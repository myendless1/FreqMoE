import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
import h5py
from accelerate import Accelerator
from tqdm import tqdm
import time
from clearml import Task
from typing import Dict, List, Tuple
import math
from original_fno import FNO2d
from FreqMoE_router import GatedFreqMoE, upcycle_fno
from torchvision.transforms import v2
import os
import lmdb
import pickle
from torch.optim.lr_scheduler import LambdaLR


class CachedCFDDataset(Dataset):
    def __init__(self, data_path, stats_path, cache_dir, is_train=True, force_preprocess=False):
        """
        Initialize the CFD dataset with LMDB caching

        Args:
            data_path: Path to the HDF5 data file
            stats_path: Path to the statistics file
            cache_dir: Directory to store the LMDB cache
            is_train: Whether this is training set
            force_preprocess: Whether to force preprocessing even if cache exists
        """
        super().__init__()
        self.data_path = data_path
        self.stats_path = stats_path
        self.is_train = is_train
        self.cache_dir = os.path.join(cache_dir, 'train' if is_train else 'test')
        os.makedirs(self.cache_dir, exist_ok=True)

        # Load the stats and create transforms
        self.transforms = self._load_transforms()

        # Initialize LMDB environment
        self.lmdb_path = os.path.join(self.cache_dir, 'data.lmdb')
        if force_preprocess or not os.path.exists(self.lmdb_path):
            self._preprocess_data()

        # Open LMDB environment
        self.env = lmdb.open(self.lmdb_path, readonly=True, lock=False)
        with self.env.begin() as txn:
            self.length = pickle.loads(txn.get(b'length'))
            # Load grid only once
            self.grid = pickle.loads(txn.get(b'grid'))

    def _load_transforms(self):
        """Load statistics and create normalizing transforms"""
        with open(self.stats_path, "r") as file:
            lines = file.readlines()
            vx_mean, vx_std = float(lines[0].split()[2].strip(',')), float(lines[0].split()[4])
            vy_mean, vy_std = float(lines[1].split()[2].strip(',')), float(lines[1].split()[4])
            density_mean, density_std = float(lines[2].split()[2].strip(',')), float(lines[2].split()[4])
            pressure_mean, pressure_std = float(lines[3].split()[2].strip(',')), float(lines[3].split()[4])

        return {
            'Vx': v2.Normalize(mean=[vx_mean], std=[vx_std]),
            'Vy': v2.Normalize(mean=[vy_mean], std=[vy_std]),
            'density': v2.Normalize(mean=[density_mean], std=[density_std]),
            'pressure': v2.Normalize(mean=[pressure_mean], std=[pressure_std])
        }

    def _create_grid(self, x_coords, y_coords):
        """Create the grid coordinates."""
        x_mesh, y_mesh = torch.meshgrid(x_coords, y_coords, indexing='ij')
        return torch.stack([x_mesh, y_mesh], dim=-1)

    def _preprocess_data(self):
        """Preprocess data and store in LMDB"""
        print(f"Preprocessing data and creating LMDB at {self.lmdb_path}")

        map_size = 1024 * 1024 * 1024 * 500  # 500GB should be enough
        env = lmdb.open(self.lmdb_path, map_size=map_size)

        with h5py.File(self.data_path, 'r', swmr=True) as f:
            # Get basic information
            sample_count = f['Vx'].shape[0]
            variables = ['Vx', 'Vy', 'density', 'pressure']

            # Create grid and store it
            x_coords = torch.from_numpy(f['x-coordinate'][:]).float()
            y_coords = torch.from_numpy(f['y-coordinate'][:]).float()
            grid = self._create_grid(x_coords, y_coords)

            # Determine indices based on train/test split
            indices = np.arange(sample_count)
            test_num = int(0.1 * sample_count)
            if not self.is_train:
                indices = indices[-test_num:]
            else:
                indices = indices[:-test_num]

            with env.begin(write=True) as txn:
                # Store length and grid
                txn.put(b'length', pickle.dumps(len(indices)))
                txn.put(b'grid', pickle.dumps(grid))

                # Process each sample
                for idx, sample_idx in enumerate(tqdm(indices)):
                    # Process input sequence (first 20 timesteps)
                    input_data = []
                    for var in variables:
                        data = f[var][sample_idx, :20]  # Take first 20 timesteps
                        # if var in ['density', 'pressure']:
                        #     data = np.log1p(np.abs(data)) * np.sign(data)
                        data = torch.from_numpy(data).float()
                        # Apply normalization for each timestep
                        normalized_data = torch.stack([
                            self.transforms[var](d.unsqueeze(0)).squeeze(0)
                            for d in data
                        ])
                        input_data.append(normalized_data)

                    x = torch.stack(input_data, dim=1)  # [20, 4, H, W]
                    x = x.permute(2, 3, 0, 1)  # [H, W, 20, 4]

                    # Process target (last timestep)
                    target_data = []
                    for var in variables:
                        data = f[var][sample_idx, -1]  # Take last timestep
                        if var in ['density', 'pressure']:
                            data = np.log1p(np.abs(data)) * np.sign(data)
                        data = torch.from_numpy(data).float()
                        data = self.transforms[var](data.unsqueeze(0)).squeeze(0)
                        target_data.append(data)

                    y = torch.stack(target_data, dim=0)  # [4, H, W]
                    y = y.permute(1, 2, 0)  # [H, W, 4]

                    # Store the preprocessed data
                    txn.put(f'{idx}_input'.encode(), pickle.dumps(x))
                    txn.put(f'{idx}_target'.encode(), pickle.dumps(y))

        env.close()
        print("Preprocessing complete")

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        """Get a sample from the dataset"""
        with self.env.begin() as txn:
            x = pickle.loads(txn.get(f'{idx}_input'.encode()))
            y = pickle.loads(txn.get(f'{idx}_target'.encode()))
        return x, self.grid, y

    def __del__(self):
        """Cleanup LMDB environment"""
        if hasattr(self, 'env'):
            self.env.close()


class CFD2DDatasetTV(torch.utils.data.Dataset):
    def __init__(self, data_path, stats_path=None, is_train=False):
        super().__init__()
        self.data_path = data_path
        self.stats_path = stats_path

        # Load the stats
        with open(stats_path, "r") as file:
            lines = file.readlines()
            vx_mean, vx_std = float(lines[0].split()[2].strip(',')), float(lines[0].split()[4])
            vy_mean, vy_std = float(lines[1].split()[2].strip(',')), float(lines[1].split()[4])
            density_mean, density_std = float(lines[2].split()[2].strip(',')), float(lines[2].split()[4])
            pressure_mean, pressure_std = float(lines[3].split()[2].strip(',')), float(lines[3].split()[4])

        # Initialize the transforms
        self.vx_transform = v2.Normalize(mean=[vx_mean], std=[vx_std])
        self.vy_transform = v2.Normalize(mean=[vy_mean], std=[vy_std])
        self.density_transform = v2.Normalize(mean=[density_mean], std=[density_std])
        self.pressure_transform = v2.Normalize(mean=[pressure_mean], std=[pressure_std])

        # Load the data
        self.f = h5py.File(self.data_path, 'r', swmr=True)
        self.sample_count = self.f['Vx'].shape[0]
        self.time_steps = self.f['Vx'].shape[1]
        self.variables = ['Vx', 'Vy', 'density', 'pressure']
        self.indexes = np.arange(self.sample_count)

        # Load coordinates (only need to do this once)
        self.x_coords = torch.from_numpy(self.f['x-coordinate'][:]).float()
        self.y_coords = torch.from_numpy(self.f['y-coordinate'][:]).float()

        # Create grid
        self.grid = self._create_grid()

        # If test, only load the last 1000 samples
        if not is_train:
            self.indexes = self.indexes[-100:]
        else:
            self.indexes = self.indexes[:-100]
        print(f"Loaded {len(self.indexes)} samples")

    def _create_grid(self) -> torch.Tensor:
        """Create the grid coordinates."""
        x_mesh, y_mesh = torch.meshgrid(self.x_coords, self.y_coords, indexing='ij')
        return torch.stack([x_mesh, y_mesh], dim=-1)

    def __len__(self):
        return len(self.indexes)

    def __getitem__(self, idx):
        sample_idx = self.indexes[idx]
        sample = []

        # Get the first 20 timesteps for input
        for var in self.variables:
            data = self.f[var][sample_idx, :20]  # Take first 20 timesteps
            if var == 'Vx':
                data = self.vx_transform(data)
            elif var == 'Vy':
                data = self.vy_transform(data)
            elif var == 'density':
                # data = np.log1p(np.abs(data)) * np.sign(data)
                data = self.density_transform(data)
            elif var == 'pressure':
                # data = np.log1p(np.abs(data)) * np.sign(data)
                data = self.pressure_transform(data)
            sample.append(torch.from_numpy(data))

        # Stack all variables and timesteps
        x = torch.stack(sample, dim=1)  # [20, 4, H, W]
        x = x.permute(2, 3, 0, 1)  # [H, W, 20, 4]

        # Get the last timestep for target
        target = []
        for var in self.variables:
            data = self.f[var][sample_idx, -1]  # Take last timestep
            if var == 'Vx':
                data = self.vx_transform(data)
            elif var == 'Vy':
                data = self.vy_transform(data)
            elif var == 'density':
                data = np.log1p(np.abs(data)) * np.sign(data)
                data = self.density_transform(data)
            elif var == 'pressure':
                data = np.log1p(np.abs(data)) * np.sign(data)
                data = self.pressure_transform(data)
            target.append(torch.from_numpy(data))

        y = torch.stack(target, dim=0)  # [4, H, W]
        y = y.permute(1, 2, 0)  # [H, W, 4]

        return x, self.grid, y


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: LambdaLR,
    accelerator: Accelerator,
    epoch: int,
    sparsity_weight: float = 0.01,
) -> Tuple[float, float, float]:
    """
    Training epoch with added sparsity loss tracking
    """
    model.train()
    total_loss = 0
    total_recon_loss = 0
    total_sparsity_loss = 0

    with tqdm(train_loader, disable=not accelerator.is_local_main_process) as pbar:
        for batch_idx, (data, grid, target) in enumerate(pbar):
            optimizer.zero_grad()
            B, H, W, T, C = data.shape

            output, sparsity_loss = model(data.view(B, H, W, -1), grid)

            # Calculate reconstruction loss
            recon_loss = F.mse_loss(output.view(B, H, W, -1), target)

            # Calculate sparsity loss
            # sparsity_loss = model.get_sparsity_loss(data)

            # Combine losses
            loss = recon_loss + sparsity_weight * sparsity_loss

            accelerator.backward(loss)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_sparsity_loss += sparsity_loss.item()

            pbar.set_description(
                f"Epoch {epoch} Train Loss: {loss.item():.6f} "
                f"(Recon: {recon_loss.item():.6f}, Sparsity: {sparsity_loss.item():.6f})"
            )

    num_batches = len(train_loader)
    return (total_loss / num_batches,
            total_recon_loss / num_batches,
            total_sparsity_loss / num_batches)


def evaluate(
    model: nn.Module,
    test_loader: DataLoader,
    accelerator: Accelerator,
    num_active_experts: int = None,
) -> Tuple[float, Dict[str, float], float]:
    """
    Evaluation function with expert usage tracking
    """
    model.eval()
    total_loss = 0
    all_l2re = {
        'Vx': [], 'Vy': [], 'density': [], 'pressure': []
    }

    # Get denormalization parameters from the dataset
    dataset = test_loader.dataset
    denorm_params = {
        'Vx': (dataset.transforms['Vx'].mean[0], dataset.transforms['Vx'].std[0]),
        'Vy': (dataset.transforms['Vy'].mean[0], dataset.transforms['Vy'].std[0]),
        'density': (dataset.transforms['density'].mean[0], dataset.transforms['density'].std[0]),
        'pressure': (dataset.transforms['pressure'].mean[0], dataset.transforms['pressure'].std[0])
    }

    print("Validating...")
    with torch.no_grad():
        for data, grid, target in tqdm(test_loader):
            B, H, W, T, C = data.shape
            output, sparsity_loss = model(data.view(B, H, W, -1), grid, num_active_experts)
            loss = F.mse_loss(output.view(B, H, W, -1), target)
            total_loss += loss.item()

            # Calculate L2RE for each variable
            for i, var in enumerate(['Vx', 'Vy', 'density', 'pressure']):
                pred = output.view(B, H, W, -1)[..., i]
                true = target.view(B, H, W, -1)[..., i]

                # Denormalize predictions and true values
                mean, std = denorm_params[var]
                pred_denorm = pred * std + mean
                true_denorm = true * std + mean

                l2_diff = torch.norm(pred_denorm - true_denorm)
                l2_true = torch.norm(true_denorm)
                l2re = l2_diff / l2_true
                all_l2re[var].append(l2re.item())

    # Get active expert ratio
    active_ratio = num_active_experts / model.num_experts

    # Average the metrics
    avg_loss = total_loss / len(test_loader)
    avg_l2re = {var: np.mean(values) for var, values in all_l2re.items()}

    return avg_loss, avg_l2re, active_ratio


def main(
    train_path: str,
    test_path: str,
    stats_path: str,
    modes: int,
    width: int,
    batch_size: int,
    num_epochs: int,
    target_modes: int,
    learning_rate: float,
    sparsity_weight: float,
    ckpt_path: str,
    project_name: str,
    cache_dir: str,
    task_name: str,
    experts_active_ratio_test=1/3,
    gate_on_data=False,
    debug=False,
):
    # Initialize ClearML
    if not debug:
        task = Task.init(project_name=project_name, task_name=task_name)
        task.connect({
            'modes': modes,
            'batch_size': batch_size,
            'num_epochs': num_epochs,
            'learning_rate': learning_rate,
            'target_modes': target_modes,
            'sparsity_weight': sparsity_weight,
        })
    else:
        task = None

    # Initialize accelerator
    accelerator = Accelerator()

    # Create datasets and dataloaders
    train_dataset = CachedCFDDataset(train_path, stats_path, cache_dir=cache_dir, is_train=True)
    test_dataset = CachedCFDDataset(test_path, stats_path, cache_dir=cache_dir, is_train=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8,
                              prefetch_factor=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=8,
                             pin_memory=True)

    # load FNO checkpoint
    if ckpt_path is not None:
        print(f"Loading FNO checkpoint from {ckpt_path}")
        checkpoint = torch.load(ckpt_path)

        fno_model = FNO2d(num_channels=4,
                          modes1=modes,
                          modes2=modes,
                          width=width,
                          initial_step=20)
        fno_model.load_state_dict(checkpoint['model'])
    else:
        fno_model = None

    # Create model and optimizer
    num_experts = (target_modes//modes) ** 2 - 1
    model = GatedFreqMoE(
        num_channels=4,
        width=width,
        initial_step=20,
        modes1=modes,
        modes2=modes,
        rank=4,
        scaling=0.1,
        num_experts=num_experts,
        gate_on_data=gate_on_data,
    )

    # do Upcycling

    model = upcycle_fno(fno_model, model) if fno_model is not None else model

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    def warmup_cosine_annealing_scheduler(optimizer, warmup_steps, total_steps, min_lr_ratio=0.1, stable_ratio=0.5):
        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            elif current_step <= total_steps * (1 - stable_ratio):  # we use cosine annealing
                progress = float(current_step - warmup_steps) / float(max(1, total_steps / 2 - warmup_steps))
                return max(min_lr_ratio, 0.5 * (1.0 + math.cos(math.pi * progress)))
            else:  # we use min_lr_ratio, the min_lr_ratio shall be cautiously considered
                return min_lr_ratio

        return LambdaLR(optimizer, lr_lambda)

    scheduler = warmup_cosine_annealing_scheduler(
        optimizer,
        warmup_steps=50,
        total_steps=len(train_loader) * num_epochs,
        min_lr_ratio=0.05,
        stable_ratio=0.3,
    )

    # Prepare for distributed training
    train_loader, test_loader, model, optimizer, scheduler = accelerator.prepare(
        train_loader, test_loader, model, optimizer, scheduler
    )

    # Training loop
    best_test_loss = float('inf')
    for epoch in range(num_epochs):
        train_loss, train_recon_loss, train_sparsity_loss = train_epoch(
            model, train_loader, optimizer, scheduler, accelerator, epoch, sparsity_weight
        )

        test_loss, l2re_metrics, active_ratio = evaluate(
            model, test_loader, accelerator, num_active_experts=int(num_experts * experts_active_ratio_test)
        )

        # Log metrics
        if accelerator.is_local_main_process:
            if task is not None:
                task.logger.report_scalar("Loss/Train/Total", "Value", train_loss, epoch)
                task.logger.report_scalar("Loss/Train/Reconstruction", "Value", train_recon_loss, epoch)
                task.logger.report_scalar("Loss/Train/Sparsity", "Value", train_sparsity_loss, epoch)
                task.logger.report_scalar("Loss/Test", "Value", test_loss, epoch)
                task.logger.report_scalar("Model/Active_Expert_Ratio", "Value", active_ratio, epoch)

                for var, value in l2re_metrics.items():
                    task.logger.report_scalar("L2RE", var, value, epoch)

            # Save best model
            if test_loss < best_test_loss:
                best_test_loss = test_loss
                if accelerator.is_local_main_process:
                    unwrapped_model = accelerator.unwrap_model(model)
                    target_path = "ckpts/{}/best_model.pt".format(task_name)
                    os.makedirs(os.path.dirname(target_path), exist_ok=True)

                    torch.save(
                        {
                            'model': unwrapped_model.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'test_loss': test_loss,
                            'epoch': epoch,
                            'active_ratio': active_ratio,
                        },
                        target_path
                    )

        accelerator.print(
            f"Epoch {epoch}:\n"
            f"  Train Loss = {train_loss:.6f} (Recon: {train_recon_loss:.6f}, "
            f"Sparsity: {train_sparsity_loss:.6f})\n"
            f"  Test Loss = {test_loss:.6f}\n"
            f"  Active Expert Ratio = {active_ratio:.2%}"
        )
        for var, value in l2re_metrics.items():
            accelerator.print(f"  {var} L2RE = {value:.6f}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-path", type=str, required=True)
    parser.add_argument("--test-path", type=str, required=True)
    parser.add_argument("--stats-path", type=str, required=True)
    parser.add_argument("--cache-dir", type=str, required=True)
    parser.add_argument("--modes", type=int, default=8)
    parser.add_argument("--target-modes", type=int, default=32)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-epochs", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--ckpt-path", type=str, default=None)
    parser.add_argument("--sparsity-weight", type=float, default=0.01)
    parser.add_argument("--project-name", type=str, default="FNO-CFD")
    parser.add_argument("--task-name", type=str, default="training")
    parser.add_argument("--experts-active-ratio-test", type=float, default=1/3)
    parser.add_argument("--gate-on-data", action='store_true')
    parser.add_argument("--debug", action='store_true')

    args = parser.parse_args()
    if args.debug:
        import debugpy
        debugpy.listen(5678)
        print("Waiting for debugger attach")
        debugpy.wait_for_client()
    main(**vars(args))
