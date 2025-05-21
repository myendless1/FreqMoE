import torch
import torch.nn as nn
import torch.nn.functional as F
from original_fno import FNO2d
import math

class LoRALayer(nn.Module):
    def __init__(self, in_dim, out_dim, rank=4):
        super().__init__()
        self.down = nn.Parameter(torch.randn(rank, in_dim) * 0.02)
        self.up = nn.Parameter(torch.randn(out_dim, rank) * 0.02)
        self.scale = 1.0

    def forward(self, x):
        return torch.matmul(torch.matmul(x, self.down.T), self.up.T) * self.scale


class FreqExpert(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, rank=4, scaling=0.1):
        super().__init__()
        self.lora_a = nn.Parameter(
            torch.randn(rank, in_channels, dtype=torch.cfloat) * 0.02
        )

        self.lora_b =nn.Parameter(
            torch.randn(out_channels, rank, modes1, modes2, dtype=torch.cfloat) * 0.02
        )

        self.scaling = scaling
    
    def get_lora_weights(self):
        """
        lora_A: [rank, in_channels]
        lora_B: [out_channels, rank, modes1, modes2]
        目标输出: [in_channels, out_channels, modes1, modes2]
        """
        # [rank, in_channels] -> [rank, in_channels, 1, 1]
        A_expanded = self.lora_a.unsqueeze(-1).unsqueeze(-1)
        
        # 矩阵乘法并重排维度
        # [out_channels, rank, modes1, modes2] * [rank, in_channels, 1, 1]
        # -> [out_channels, in_channels, modes1, modes2]
        lora_weights = torch.sum(self.lora_b.unsqueeze(2) * A_expanded.unsqueeze(0), dim=1)
        
        # 转置得到 [in_channels, out_channels, modes1, modes2]
        return lora_weights.permute(1, 0, 2, 3) * self.scaling

class SpectralConv2d_fast_LoRA_MoE(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, rank=4, scaling=0.1, num_experts=15):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        self.rank = rank
        self.num_experts = num_experts

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

        # initialize the experts
        self.experts_weights1 = nn.ModuleList(
            [
                FreqExpert(in_channels, out_channels, modes1, modes2, rank=rank, scaling=scaling)
                for _ in range(num_experts)
            ]
        )

        self.experts_weights2 = nn.ModuleList(
            [
                FreqExpert(in_channels, out_channels, modes1, modes2, rank=rank, scaling=scaling)
                for _ in range(num_experts)
            ]
        )
    
    def compl_mul2d(self, input, weights):
        return torch.einsum("bixy,ioxy->boxy", input, weights)
    

    def forward(self, x):
        batchsize = x.shape[0]
        x_ft = torch.fft.rfft2(x)

        # Initialize out_ft
        out_ft = torch.zeros(
            batchsize,
            self.out_channels,
            x.size(-2),
            x.size(-1) // 2 + 1,
            dtype=torch.cfloat,
            device=x.device,
        )

        # compute first modes region with raw weights
        out_ft[:, :, :self.modes1, :self.modes2] = self.compl_mul2d(
            x_ft[:, :, :self.modes1, :self.modes2], self.weights1
        )

        out_ft[:, :, -self.modes1:, :self.modes2] = self.compl_mul2d(
            x_ft[:, :, -self.modes1:, :self.modes2], self.weights2
        )

        # compute other modes region with MoE weights, We let each expert to take resposible for one region of modes
        bound = out_ft.shape[3]
        for i in range(self.num_experts):
            
            expert_region_size = int(math.sqrt(self.num_experts+1))
            row_idx = int((i+1) // expert_region_size)
            col_idx = int((i+1) % expert_region_size)
            
            row_idx_start = self.modes1*(row_idx)
            row_idx_end = self.modes1*(row_idx+1)

            col_idx_start = self.modes2*(col_idx)
            col_idx_end = self.modes2*(col_idx+1)

            out_ft[:, :, row_idx_start:row_idx_end, col_idx_start:col_idx_end] = self.compl_mul2d(
                x_ft[:, :, row_idx_start:row_idx_end, col_idx_start:col_idx_end], self.experts_weights1[i].get_lora_weights()
            )

            out_ft[:, :, bound-row_idx_end:bound-row_idx_start, col_idx_start:col_idx_end] = self.compl_mul2d(
                x_ft[:, :, bound-row_idx_end:bound-row_idx_start, col_idx_start:col_idx_end], self.experts_weights2[i].get_lora_weights()
            )
        

        return torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))



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
                freq_moe_model: FreqMoE, 
                strict: bool = True) -> None:
    """
    Transfer weights from a trained FNO2d model to a FreqMoE model.
    
    Args:
        fno_model: Trained FNO2d model
        freq_moe_model: Initialized FreqMoE model
        strict: Whether to require exact match of architectures
    """
    # 首先确保基础架构参数匹配
    assert fno_model.width == freq_moe_model.width, "Model widths do not match"
    assert fno_model.modes1 == freq_moe_model.modes1, "Modes1 do not match"
    assert fno_model.modes2 == freq_moe_model.modes2, "Modes2 do not match"
    
    # 复制非专家层的权重
    freq_moe_model.fc0.weight.data = fno_model.fc0.weight.data.clone()
    freq_moe_model.fc0.bias.data = fno_model.fc0.bias.data.clone()
    
    freq_moe_model.fc1.weight.data = fno_model.fc1.weight.data.clone()
    freq_moe_model.fc1.bias.data = fno_model.fc1.bias.data.clone()
    
    freq_moe_model.fc2.weight.data = fno_model.fc2.weight.data.clone()
    freq_moe_model.fc2.bias.data = fno_model.fc2.bias.data.clone()
    
    # 复制1x1卷积层权重
    for i in range(4):
        getattr(freq_moe_model, f'w{i}').weight.data = getattr(fno_model, f'w{i}').weight.data.clone()
        getattr(freq_moe_model, f'w{i}').bias.data = getattr(fno_model, f'w{i}').bias.data.clone()
    
    # 复制主要权重到专家层的基础权重
    spectral_layers = [(freq_moe_model.conv0, fno_model.conv0),
                      (freq_moe_model.conv1, fno_model.conv1),
                      (freq_moe_model.conv2, fno_model.conv2),
                      (freq_moe_model.conv3, fno_model.conv3)]
    
    for moe_conv, fno_conv in spectral_layers:
        # 复制基础权重
        moe_conv.weights1.data = fno_conv.weights1.data.clone()
        moe_conv.weights2.data = fno_conv.weights2.data.clone()
        
        # Experts的LoRA权重保持随机初始化
        # 你可以选择在这里自定义experts的初始化策略
        for expert in moe_conv.experts_weights1:
            nn.init.normal_(expert.lora_a, mean=0.0, std=0.02)
            nn.init.normal_(expert.lora_b, mean=0.0, std=0.02)
        
        for expert in moe_conv.experts_weights2:
            nn.init.normal_(expert.lora_a, mean=0.0, std=0.02)
            nn.init.normal_(expert.lora_b, mean=0.0, std=0.02)

def test_upcycle_fno():
    """
    Test function to verify the weight transfer functionality
    """
    # 创建测试模型
    num_channels = 4
    modes = 8
    width = 32
    initial_step = 20
    
    # 初始化原始FNO模型
    fno_model = FNO2d(num_channels=num_channels, 
                      modes1=modes, 
                      modes2=modes, 
                      width=width, 
                      initial_step=initial_step)

    # Fully Modes FNO Model
    fno_model_full = FNO2d(num_channels=num_channels, 
                      modes1=modes*8, 
                      modes2=modes*8, 
                      width=width, 
                      initial_step=initial_step) 
    
    # 初始化FreqMoE模型
    freq_moe_model = FreqMoE(num_channels=num_channels,
                            modes1=modes,
                            modes2=modes,
                            width=width,
                            initial_step=initial_step,
                            rank=4,
                            scaling=0.1,
                            num_experts=63)
    
    # get the number of parameters
    total_params = sum(p.numel() for p in fno_model.parameters())
    print(f"Total number of parameters in FNO model: {total_params}")

    total_params = sum(p.numel() for p in fno_model_full.parameters())
    print(f"Total number of parameters in FNO_full model: {total_params}")
    
    total_params = sum(p.numel() for p in freq_moe_model.parameters())
    print(f"Total number of parameters in FreqMoE model: {total_params}")
    
    # 生成一些随机输入数据进行测试
    batch_size = 2
    size = 512
    x = torch.randn(batch_size, size, size, initial_step * num_channels)
    grid = torch.randn(batch_size, size, size, 2)
    
    # 记录原始FNO输出
    fno_output = fno_model(x, grid)
    
    
    # 执行权重迁移
    upcycle_fno(fno_model, freq_moe_model)
    
    # 验证非LoRA权重是否正确复制
    assert torch.allclose(fno_model.fc0.weight.data, 
                         freq_moe_model.fc0.weight.data), "FC0 weights mismatch"
    
    # 验证1x1卷积层权重
    assert torch.allclose(fno_model.w0.weight.data,
                         freq_moe_model.w0.weight.data), "Conv weight mismatch"
    
    # 验证谱卷积层的基础权重
    assert torch.allclose(fno_model.conv0.weights1.data,
                         freq_moe_model.conv0.weights1.data), "Spectral conv weights mismatch"
    
    # 运行FreqMoE模型并比较基础输出
    # 注意：由于LoRA初始化是随机的，输出会有细微差异
    freq_moe_output = freq_moe_model(x, grid)
    
    print("Weight transfer test completed successfully!")
    print(f"Output shapes - FNO: {fno_output.shape}, FreqMoE: {freq_moe_output.shape}")
    
    # 计算输出差异（应该很小，因为LoRA初始化的scale很小）
    diff = torch.mean(torch.abs(fno_output - freq_moe_output))
    print(f"Mean absolute difference between outputs: {diff:.6f}")



if __name__ == "__main__":
    test_upcycle_fno()


















def convert_spectralconv_to_lora(original_layer, rank=4, freeze_original=True):
    lora_layer = SpectralConv2d_fast_LoRA(
        original_layer.in_channels,
        original_layer.out_channels,
        original_layer.modes1,
        original_layer.modes2,
        rank=rank,
        freeze_original=freeze_original
    )
    
    # Copy original weights
    lora_layer.weights1.data = original_layer.weights1.data.clone()
    lora_layer.weights2.data = original_layer.weights2.data.clone()
    
    return lora_layer