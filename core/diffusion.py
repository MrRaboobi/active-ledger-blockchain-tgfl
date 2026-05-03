import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from diffusers import UNet1DModel, DDPMScheduler
from pathlib import Path

# Default path for pre-trained diffusion weights
PRETRAINED_PATH = Path("checkpoints/diffusion_pretrained.pth")

class ECGDiffusionGenerator:
    def __init__(self, pretrained_path=None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.scheduler = DDPMScheduler(num_train_timesteps=1000)
        self.unet = UNet1DModel(
            sample_size=384,  # Padded from 360 to be divisible by 16 (4 downsampling blocks)
            in_channels=2,
            out_channels=1,
            down_block_types=("DownBlock1D", "DownBlock1D", "AttnDownBlock1D", "AttnDownBlock1D"),
            up_block_types=("AttnUpBlock1D", "AttnUpBlock1D", "UpBlock1D", "UpBlock1D"),
            block_out_channels=(32, 64, 128, 256)
        ).to(self.device)

        # Auto-load pre-trained weights if available
        if pretrained_path is None:
            pretrained_path = PRETRAINED_PATH
        if Path(pretrained_path).exists():
            self.load_weights(pretrained_path)
            print(f"[DIFFUSION] Loaded pre-trained weights from {pretrained_path}")

    def generate_synthetic_ecg(self, class_label: int, quantity: int, num_inference_steps: int = 50) -> np.ndarray:
        noisy_ecg = torch.randn((quantity, 1, 384)).to(self.device)
        lbl_channel = torch.full((quantity, 1, 384), class_label / 4.0).to(self.device)
        
        self.scheduler.set_timesteps(num_inference_steps)
        
        for t in self.scheduler.timesteps:
            with torch.no_grad():
                model_input = torch.cat([noisy_ecg, lbl_channel], dim=1)
                pred_noise = self.unet(model_input, t).sample
                noisy_ecg = self.scheduler.step(pred_noise, t, noisy_ecg).prev_sample
                
        # Crop back to original sequence length of 360
        return noisy_ecg[:, :, :360].cpu().squeeze(1).numpy()

    # ── Training methods ──────────────────────────────────────────────────

    def train_on_data(self, ecg_samples: np.ndarray, labels: np.ndarray, epochs: int = 30,
                      batch_size: int = 32, lr: float = 1e-4, device: str = None):
        """
        Train the UNet via standard DDPM noise-prediction objective, conditioned on class labels.

        Args:
            ecg_samples: numpy array of shape (N, 360) — real ECG signals
            labels: numpy array of shape (N,) — integer class labels
            epochs: number of training epochs
            batch_size: training batch size
            lr: learning rate
            device: override device (e.g. 'cuda' for GPU acceleration)
        """
        train_device = device or self.device

        # Move UNet to training device
        self.unet = self.unet.to(train_device)

        # Pad 360→384 and add channel dim → (N, 1, 384)
        padded = np.pad(ecg_samples, ((0, 0), (0, 24)), mode='constant')
        tensor_data = torch.FloatTensor(padded).unsqueeze(1)  # (N, 1, 384)
        
        # Create label channel: (N, 1, 384) — normalized to 0-1 range
        labels_tensor = torch.FloatTensor(labels).view(-1, 1, 1).expand(-1, 1, 384) / 4.0
        
        # Combine into (N, 2, 384)
        combined_data = torch.cat([tensor_data, labels_tensor], dim=1)

        dataset = torch.utils.data.TensorDataset(combined_data)
        loader  = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        optimizer = torch.optim.AdamW(self.unet.parameters(), lr=lr)
        noise_scheduler = DDPMScheduler(num_train_timesteps=1000)

        self.unet.train()
        for epoch in range(epochs):
            epoch_loss = 0.0
            n_batches  = 0

            for (batch,) in loader:
                batch = batch.to(train_device)
                
                ecg_channel = batch[:, 0:1, :]
                lbl_channel = batch[:, 1:2, :]

                # Sample random timesteps
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps,
                    (batch.shape[0],), device=train_device
                ).long()

                # Add noise ONLY to the ECG channel (labels stay clean)
                noise = torch.randn_like(ecg_channel)
                noisy_ecg = noise_scheduler.add_noise(ecg_channel, noise, timesteps)
                
                # Re-concatenate noisy ECG with clean label channel
                noisy_input = torch.cat([noisy_ecg, lbl_channel], dim=1)

                # Predict noise
                pred = self.unet(noisy_input, timesteps).sample

                # MSE loss
                loss = nn.functional.mse_loss(pred, noise)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                n_batches  += 1

            avg_loss = epoch_loss / max(n_batches, 1)
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"  [DIFFUSION] Epoch {epoch+1}/{epochs}  Loss: {avg_loss:.6f}")

        # Move back to original device after training
        self.unet = self.unet.to(self.device)
        print(f"[DIFFUSION] Training complete. Final loss: {avg_loss:.6f}")

    def save_weights(self, path=None):
        """Save UNet weights to disk."""
        if path is None:
            path = PRETRAINED_PATH
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.unet.state_dict(), path)
        print(f"[DIFFUSION] Weights saved to {path}")

    def load_weights(self, path=None):
        """Load UNet weights from disk."""
        if path is None:
            path = PRETRAINED_PATH
        state = torch.load(path, map_location=self.device, weights_only=True)
        self.unet.load_state_dict(state)


if __name__ == "__main__":
    generator = ECGDiffusionGenerator()
    print("Generating simulated untrained 1D LDM ECG samples...")
    synthetic_samples = generator.generate_synthetic_ecg(class_label=2, quantity=1, num_inference_steps=50)
    
    sample = synthetic_samples[0]
    
    plt.figure(figsize=(10, 4))
    plt.plot(sample, color='blue', alpha=0.8)
    plt.title("Simulated 1D Latent Diffusion ECG (Untrained)")
    plt.xlabel("Time steps")
    plt.ylabel("Normalized Amplitude")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig("synthetic_ecg_sample.pdf")
    plt.close()
    
    print(f"Shape of synthetic generated data: {synthetic_samples.shape}")
    print("PDF saved as synthetic_ecg_sample.pdf in root directory.")
