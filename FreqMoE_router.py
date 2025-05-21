import torch
import torch.nn as nn
import torch.nn.functional as F
from original_fno import FNO2d
import math


class GatedFreqExpert(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, rank=4, scaling=0.1, gate_on_data=False,
                 expert_index=0, expert_region_size=4):
        super().__init__()
        self.modes1, self.modes2, self.rank, self.expert_region_size = modes1, modes2, rank, expert_region_size
        self.lora_a = nn.Parameter(
            torch.randn(rank, in_channels, dtype=torch.cfloat) * 0.02
        )
        self.lora_b = nn.Parameter(
            torch.randn(out_channels, rank, modes1, modes2, dtype=torch.cfloat) * 0.02
        )
        self.scaling = scaling
        self.gate_on_data = gate_on_data
        if gate_on_data:
            self.gate = nn.Linear(modes1*modes2*in_channels*2, 1)
        else:
            self.gate = nn.Parameter(torch.zeros(1), requires_grad=True)  # Initialize to zero
        self.temperature = 1.0
        self.row_start, self.row_end, self.col_start, self.col_end = self.get_expert_region(expert_index)

    def get_expert_region(self, expert_idx):
        """Get the frequency region for a given expert index"""
        row_idx = int((expert_idx+1) // self.expert_region_size)
        col_idx = int((expert_idx+1) % self.expert_region_size)

        row_start = self.modes1 * row_idx
        row_end = self.modes1 * (row_idx + 1)
        col_start = self.modes2 * col_idx
        col_end = self.modes2 * (col_idx + 1)

        return row_start, row_end, col_start, col_end

    def get_gate_value(self, x):
        if self.gate_on_data:
            x = x[..., self.row_start:self.row_end, self.col_start:self.col_end]
            x = x.flatten(start_dim=1)
            # [B, num_experts, 1, 1]
            return torch.sigmoid(self.gate(torch.view_as_real(x).flatten(-2, -1)) / self.temperature).unsqueeze(-1).unsqueeze(-1)
        else:
            return torch.sigmoid(self.gate / self.temperature)

    def get_lora_weights(self):
        A_expanded = self.lora_a.unsqueeze(-1).unsqueeze(-1)
        lora_weights = torch.sum(self.lora_b.unsqueeze(2) * A_expanded.unsqueeze(0), dim=1)
        return lora_weights.permute(1, 0, 2, 3) * self.scaling


class SpectralConv2d_fast_LoRA_GatedMoE(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, rank=4, scaling=0.1, num_experts=15, gate_on_data=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        self.rank = rank
        self.num_experts = num_experts
        self.expert_region_size = int(math.sqrt(num_experts+1))

        # Original weights for low frequencies
        self.scale = 1 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(
                in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat
            )
        )
        self.weights2 = nn.Parameter(
            self.scale * torch.rand(
                in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat
            )
        )

        # Initialize experts
        self.experts_weights1 = nn.ModuleList([
            GatedFreqExpert(in_channels, out_channels, modes1, modes2, rank=rank, scaling=scaling, gate_on_data=gate_on_data)
            for _ in range(num_experts)
        ])
        self.experts_weights2 = nn.ModuleList([
            GatedFreqExpert(in_channels, out_channels, modes1, modes2, rank=rank, scaling=scaling, gate_on_data=gate_on_data)
            for _ in range(num_experts)
        ])

        # Cache for selected experts during inference
        self.register_buffer('selected_experts', None)
        self.training = True
        self.gate_on_data = gate_on_data

    def get_sparsity_loss(self, x):
        gate_values1 = torch.stack([expert.get_gate_value(x) for expert in self.experts_weights1])
        gate_values2 = torch.stack([expert.get_gate_value(x) for expert in self.experts_weights2])
        return torch.mean(gate_values1) + torch.mean(gate_values2)

    def select_top_k_experts(self, x_ft, k):
        """Select top-k experts based on gate values"""
        gate_values1 = torch.stack([expert.get_gate_value(x_ft) for expert in self.experts_weights1])
        gate_values2 = torch.stack([expert.get_gate_value(x_ft) for expert in self.experts_weights2])

        # Combine gate values and get indices
        gate_values = (gate_values1 + gate_values2) / 2
        _, indices = torch.topk(gate_values.squeeze(), k, dim=0)

        # Save selected expert indices
        self.selected_experts = indices.cpu()
        return indices

    def get_expert_region(self, expert_idx):
        """Get the frequency region for a given expert index"""
        row_idx = int((expert_idx+1) // self.expert_region_size)
        col_idx = int((expert_idx+1) % self.expert_region_size)

        row_start = self.modes1 * row_idx
        row_end = self.modes1 * (row_idx + 1)
        col_start = self.modes2 * col_idx
        col_end = self.modes2 * (col_idx + 1)

        return row_start, row_end, col_start, col_end

    def compl_mul2d(self, input, weights):
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x, num_active_experts=None):
        batchsize = x.shape[0]
        x_ft = torch.fft.rfft2(x)
        sploss = self.get_sparsity_loss(x_ft)  # must be placed here as the input x_ft is needed

        out_ft = torch.zeros(
            batchsize,
            self.out_channels,
            x.size(-2),
            x.size(-1) // 2 + 1,
            dtype=torch.cfloat,
            device=x.device,
        )

        # Always compute low frequency region with original weights
        out_ft[:, :, :self.modes1, :self.modes2] = self.compl_mul2d(
            x_ft[:, :, :self.modes1, :self.modes2], self.weights1
        )
        out_ft[:, :, -self.modes1:, :self.modes2] = self.compl_mul2d(
            x_ft[:, :, -self.modes1:, :self.modes2], self.weights2
        )

        bound = out_ft.shape[2]

        if self.training:
            # During training, use all experts with gates
            for expert_index in range(self.num_experts):
                row_start, row_end, col_start, col_end = self.get_expert_region(expert_index)

                # Apply gated experts
                out_ft[:, :, row_start:row_end, col_start:col_end] = self.compl_mul2d(
                    x_ft[:, :, row_start:row_end, col_start:col_end],
                    self.experts_weights1[expert_index].get_lora_weights()
                ) * self.experts_weights1[expert_index].get_gate_value(x_ft)

                out_ft[:, :, bound-row_end:bound-row_start, col_start:col_end] = self.compl_mul2d(
                    x_ft[:, :, bound-row_end:bound-row_start, col_start:col_end],
                    self.experts_weights2[expert_index].get_lora_weights()
                ) * self.experts_weights2[expert_index].get_gate_value(x_ft)
        else:
            # During inference, use only top-k experts
            if num_active_experts is None:
                num_active_experts = self.num_experts // 3  # Default to 1/3 of experts

            # Select top-k experts if not already selected
            if self.selected_experts is None:
                self.select_top_k_experts(x_ft, num_active_experts)

            # Use only selected experts
            if self.gate_on_data:
                for bs_index in range(batchsize):
                    for expert_index in self.selected_experts[:, bs_index]:
                        row_start, row_end, col_start, col_end = self.get_expert_region(expert_index.item())

                        # Apply selected experts with gate multiplication
                        out_ft[bs_index:bs_index+1, :, row_start:row_end, col_start:col_end] = self.compl_mul2d(
                            x_ft[bs_index:bs_index+1, :, row_start:row_end, col_start:col_end],
                            self.experts_weights1[expert_index].get_lora_weights()
                        ) * self.experts_weights1[expert_index].get_gate_value(x_ft[bs_index:bs_index+1])

                        out_ft[bs_index:bs_index+1, :, bound-row_end:bound-row_start, col_start:col_end] = self.compl_mul2d(
                            x_ft[bs_index:bs_index+1, :, bound-row_end:bound-row_start, col_start:col_end],
                            self.experts_weights2[expert_index].get_lora_weights()
                        ) * self.experts_weights2[expert_index].get_gate_value(x_ft[bs_index:bs_index+1])
            else:  # all data share the same gate cause the gate does not rely on data
                for expert_index in self.selected_experts:
                    row_start, row_end, col_start, col_end = self.get_expert_region(expert_index.item())

                    # Apply selected experts with gate multiplication
                    out_ft[:, :, row_start:row_end, col_start:col_end] = self.compl_mul2d(
                        x_ft[:, :, row_start:row_end, col_start:col_end],
                        self.experts_weights1[expert_index].get_lora_weights()
                    ) * self.experts_weights1[expert_index].get_gate_value(x_ft)

                    out_ft[:, :, bound-row_end:bound-row_start, col_start:col_end] = self.compl_mul2d(
                        x_ft[:, :, bound-row_end:bound-row_start, col_start:col_end],
                        self.experts_weights2[expert_index].get_lora_weights()
                    ) * self.experts_weights2[expert_index].get_gate_value(x_ft)

        return torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1))), sploss


class SpectralConv2d_fast_LoRA(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, rank=4, freeze_original=True):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        self.rank = rank

        # Original weights
        self.scale = 1 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(
                in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat
            )
        )
        self.weights2 = nn.Parameter(
            self.scale * torch.rand(
                in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat
            )
        )

        # Freeze original weights if specified
        if freeze_original:
            self.weights1.requires_grad = False
            self.weights2.requires_grad = False

        # LoRA parameters for weights1 and weights2
        # 修改维度设置以确保正确的矩阵乘法
        self.lora1_A = nn.Parameter(
            torch.randn(rank, in_channels, dtype=torch.cfloat) * 0.02
        )
        self.lora1_B = nn.Parameter(
            torch.randn(out_channels, rank, modes1, modes2, dtype=torch.cfloat) * 0.02
        )

        self.lora2_A = nn.Parameter(
            torch.randn(rank, in_channels, dtype=torch.cfloat) * 0.02
        )
        self.lora2_B = nn.Parameter(
            torch.randn(out_channels, rank, modes1, modes2, dtype=torch.cfloat) * 0.02
        )

        self.scaling = 0.1  # LoRA scaling factor

    def get_lora_weights(self, lora_A, lora_B):
        """
        lora_A: [rank, in_channels]
        lora_B: [out_channels, rank, modes1, modes2]
        目标输出: [in_channels, out_channels, modes1, modes2]
        """
        # [rank, in_channels] -> [rank, in_channels, 1, 1]
        A_expanded = lora_A.unsqueeze(-1).unsqueeze(-1)

        # 矩阵乘法并重排维度
        # [out_channels, rank, modes1, modes2] * [rank, in_channels, 1, 1]
        # -> [out_channels, in_channels, modes1, modes2]
        lora_weights = torch.sum(lora_B.unsqueeze(2) * A_expanded.unsqueeze(0), dim=1)

        # 转置得到 [in_channels, out_channels, modes1, modes2]
        return lora_weights.permute(1, 0, 2, 3) * self.scaling

    def compl_mul2d(self, input, weights):
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        x_ft = torch.fft.rfft2(x)

        # Compute LoRA contributions
        weights1_lora = self.get_lora_weights(self.lora1_A, self.lora1_B)
        weights2_lora = self.get_lora_weights(self.lora2_A, self.lora2_B)

        # Add LoRA weights to original weights
        weights1_adapted = self.weights1 + weights1_lora
        weights2_adapted = self.weights2 + weights2_lora

        out_ft = torch.zeros(
            batchsize,
            self.out_channels,
            x.size(-2),
            x.size(-1) // 2 + 1,
            dtype=torch.cfloat,
            device=x.device,
        )

        out_ft[:, :, :self.modes1, :self.modes2] = self.compl_mul2d(
            x_ft[:, :, :self.modes1, :self.modes2], weights1_adapted
        )
        out_ft[:, :, -self.modes1:, :self.modes2] = self.compl_mul2d(
            x_ft[:, :, -self.modes1:, :self.modes2], weights2_adapted
        )

        return torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))


class GatedFreqMoE(nn.Module):
    def __init__(self, num_channels, width, initial_step, modes1, modes2, rank=4, scaling=0.1, num_experts=15, gate_on_data=False):
        super().__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.padding = 2
        self.fc0 = nn.Linear(initial_step * num_channels + 2, self.width)
        self.num_experts = num_experts

        # Spectral convolutions with gated experts
        self.conv0 = SpectralConv2d_fast_LoRA_GatedMoE(
            self.width, self.width, self.modes1, self.modes2, rank=rank, scaling=scaling, num_experts=num_experts, gate_on_data=gate_on_data)
        self.conv1 = SpectralConv2d_fast_LoRA_GatedMoE(
            self.width, self.width, self.modes1, self.modes2, rank=rank, scaling=scaling, num_experts=num_experts, gate_on_data=gate_on_data)
        self.conv2 = SpectralConv2d_fast_LoRA_GatedMoE(
            self.width, self.width, self.modes1, self.modes2, rank=rank, scaling=scaling, num_experts=num_experts, gate_on_data=gate_on_data)
        self.conv3 = SpectralConv2d_fast_LoRA_GatedMoE(
            self.width, self.width, self.modes1, self.modes2, rank=rank, scaling=scaling, num_experts=num_experts, gate_on_data=gate_on_data)

        self.w0 = nn.Conv2d(self.width, self.width, 1)
        self.w1 = nn.Conv2d(self.width, self.width, 1)
        self.w2 = nn.Conv2d(self.width, self.width, 1)
        self.w3 = nn.Conv2d(self.width, self.width, 1)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, num_channels)

    # def get_sparsity_loss(self, x):
    #     return (self.conv0.get_sparsity_loss(x) +
    #             self.conv1.get_sparsity_loss(x) +
    #             self.conv2.get_sparsity_loss(x) +
    #             self.conv3.get_sparsity_loss(x))

    def train(self, mode=True):
        """Override train mode to handle expert selection"""
        super().train(mode)
        if not mode:  # If switching to eval mode
            # Reset selected experts to force recomputation
            for conv in [self.conv0, self.conv1, self.conv2, self.conv3]:
                conv.selected_experts = None
        return self

    def forward(self, x, grid, num_active_experts=None):
        sparsity_loss = 0
        x = torch.cat((x, grid), dim=-1)
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)

        x = F.pad(x, [0, self.padding, 0, self.padding])

        x1, sploss = self.conv0(x, num_active_experts)
        sparsity_loss += sploss
        x2 = self.w0(x)
        x = x1 + x2
        x = F.gelu(x)

        x1, sploss = self.conv1(x, num_active_experts)
        sparsity_loss += sploss
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(x)

        x1, sploss = self.conv2(x, num_active_experts)
        sparsity_loss += sploss
        x2 = self.w2(x)
        x = x1 + x2
        x = F.gelu(x)

        x1, sploss = self.conv3(x, num_active_experts)
        sparsity_loss += sploss
        x2 = self.w3(x)
        x = x1 + x2

        x = x[..., : -self.padding, : -self.padding]
        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)

        return x.unsqueeze(-2), sparsity_loss


class FreqMoE(nn.Module):
    def __init__(self, num_channels, width, initial_step, modes1, modes2, rank=4, scaling=0.1, num_experts=15):
        super().__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.padding = 2  # pad the domain if input is non-periodic
        self.fc0 = nn.Linear(initial_step * num_channels + 2, self.width)
        # input channel is 12: the solution of the previous 10 timesteps + 2 locations (u(t-10, x, y), ..., u(t-1, x, y),  x, y)

        self.conv0 = SpectralConv2d_fast_LoRA_MoE(
            self.width, self.width, self.modes1, self.modes2, rank=rank, scaling=scaling, num_experts=num_experts)

        self.conv1 = SpectralConv2d_fast_LoRA_MoE(
            self.width, self.width, self.modes1, self.modes2, rank=rank, scaling=scaling, num_experts=num_experts
        )
        self.conv2 = SpectralConv2d_fast_LoRA_MoE(
            self.width, self.width, self.modes1, self.modes2, rank=rank, scaling=scaling, num_experts=num_experts
        )
        self.conv3 = SpectralConv2d_fast_LoRA_MoE(
            self.width, self.width, self.modes1, self.modes2, rank=rank, scaling=scaling, num_experts=num_experts
        )
        self.w0 = nn.Conv2d(self.width, self.width, 1)
        self.w1 = nn.Conv2d(self.width, self.width, 1)
        self.w2 = nn.Conv2d(self.width, self.width, 1)
        self.w3 = nn.Conv2d(self.width, self.width, 1)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, num_channels)

    def forward(self, x, grid):
        # x dim = [b, x1, x2, t*v]

        x = torch.cat((x, grid), dim=-1)
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)

        # Pad tensor with boundary condition
        x = F.pad(x, [0, self.padding, 0, self.padding])

        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2

        x = x[..., : -self.padding, : -self.padding]  # Unpad the tensor
        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)

        return x.unsqueeze(-2)


def upcycle_fno(fno_model: FNO2d,
                gated_moe_model: GatedFreqMoE,
                strict: bool = True) -> None:
    """
    Transfer weights from a trained FNO2d model to a GatedFreqMoE model.

    Args:
        fno_model: Trained FNO2d model
        gated_moe_model: Initialized GatedFreqMoE model
        strict: Whether to require exact match of architectures
    """
    # 验证基础架构参数匹配
    assert fno_model.width == gated_moe_model.width, "Model widths do not match"
    assert fno_model.modes1 == gated_moe_model.modes1, "Modes1 do not match"
    assert fno_model.modes2 == gated_moe_model.modes2, "Modes2 do not match"

    # 复制非专家层的权重
    gated_moe_model.fc0.weight.data = fno_model.fc0.weight.data.clone()
    gated_moe_model.fc0.bias.data = fno_model.fc0.bias.data.clone()

    gated_moe_model.fc1.weight.data = fno_model.fc1.weight.data.clone()
    gated_moe_model.fc1.bias.data = fno_model.fc1.bias.data.clone()

    gated_moe_model.fc2.weight.data = fno_model.fc2.weight.data.clone()
    gated_moe_model.fc2.bias.data = fno_model.fc2.bias.data.clone()

    # 复制1x1卷积层权重
    for i in range(4):
        getattr(gated_moe_model, f'w{i}').weight.data = getattr(fno_model, f'w{i}').weight.data.clone()
        getattr(gated_moe_model, f'w{i}').bias.data = getattr(fno_model, f'w{i}').bias.data.clone()

    # 复制主要权重到专家层的基础权重
    spectral_layers = [(gated_moe_model.conv0, fno_model.conv0),
                       (gated_moe_model.conv1, fno_model.conv1),
                       (gated_moe_model.conv2, fno_model.conv2),
                       (gated_moe_model.conv3, fno_model.conv3)]

    for moe_conv, fno_conv in spectral_layers:
        # 复制基础权重
        moe_conv.weights1.data = fno_conv.weights1.data.clone()
        moe_conv.weights2.data = fno_conv.weights2.data.clone()

        # 初始化experts的LoRA权重和gates
        for expert in moe_conv.experts_weights1:
            nn.init.normal_(expert.lora_a, mean=0.0, std=0.02)
            nn.init.normal_(expert.lora_b, mean=0.0, std=0.02)
            if isinstance(expert.gate, nn.Linear):
                nn.init.zeros_(expert.gate.weight)
                nn.init.zeros_(expert.gate.bias)
            else:
                nn.init.zeros_(expert.gate)  # 初始化gate为0

        for expert in moe_conv.experts_weights2:
            nn.init.normal_(expert.lora_a, mean=0.0, std=0.02)
            nn.init.normal_(expert.lora_b, mean=0.0, std=0.02)
            if isinstance(expert.gate, nn.Linear):
                nn.init.zeros_(expert.gate.weight)
                nn.init.zeros_(expert.gate.bias)
            else:
                nn.init.zeros_(expert.gate)  # 初始化gate为0
    return gated_moe_model


def test_gated_freqmoe():
    """
    Test function to verify the GatedFreqMoE functionality
    """
    # 创建测试模型
    num_channels = 4
    modes = 8
    width = 32
    initial_step = 20
    num_experts = 15

    # 初始化模型
    fno_model = FNO2d(num_channels=num_channels,
                      modes1=modes,
                      modes2=modes,
                      width=width,
                      initial_step=initial_step)

    gated_moe_model = GatedFreqMoE(num_channels=num_channels,
                                   modes1=modes,
                                   modes2=modes,
                                   width=width,
                                   initial_step=initial_step,
                                   rank=4,
                                   scaling=0.1,
                                   num_experts=num_experts)

    # 打印参数数量
    fno_params = sum(p.numel() for p in fno_model.parameters())
    moe_params = sum(p.numel() for p in gated_moe_model.parameters())
    print(f"FNO parameters: {fno_params:,}")
    print(f"GatedFreqMoE parameters: {moe_params:,}")

    # 生成测试数据
    batch_size = 2
    size = 64  # 使用小一点的size加快测试
    x = torch.randn(batch_size, size, size, initial_step * num_channels)
    grid = torch.randn(batch_size, size, size, 2)

    # 测试权重迁移
    print("\nTesting weight transfer...")
    upcycle_fno(fno_model, gated_moe_model)

    # 验证权重迁移
    assert torch.allclose(fno_model.fc0.weight.data,
                          gated_moe_model.fc0.weight.data), "FC0 weights mismatch"
    print("Weight transfer verified successfully")

    # 测试训练模式
    print("\nTesting training mode behavior...")
    gated_moe_model.train()
    train_output = gated_moe_model(x, grid)
    train_sparsity = gated_moe_model.get_sparsity_loss(x)
    print(f"Training mode - Sparsity loss: {train_sparsity:.6f}")

    # 测试推理模式
    print("\nTesting inference mode behavior...")

    # 测试不同数量的active experts
    test_k = [5, 10]
    for k in test_k:
        gated_moe_model.eval()  # re-assign the experts
        with torch.no_grad():
            inference_output = gated_moe_model(x, grid, num_active_experts=k)

        print(f"\nTesting with {k} active experts:")
        # 检查gate values
        for conv_idx, conv in enumerate([gated_moe_model.conv0, gated_moe_model.conv1,
                                         gated_moe_model.conv2, gated_moe_model.conv3]):
            if conv.selected_experts is not None:
                active_experts = len(conv.selected_experts)
                print(f"Conv{conv_idx} - Active experts: {active_experts}")

            # 获取所有experts的gate values
            gate_values = torch.stack([expert.get_gate_value() for expert in conv.experts_weights1])
            mean_gate = torch.mean(gate_values)
            max_gate = torch.max(gate_values)
            print(f"Conv{conv_idx} - Mean gate value: {mean_gate:.4f}, Max gate value: {max_gate:.4f}")

    print("\nAll tests completed successfully!")


if __name__ == "__main__":
    test_gated_freqmoe()
