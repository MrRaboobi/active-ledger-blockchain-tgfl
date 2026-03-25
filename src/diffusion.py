import torch
import numpy as np
import matplotlib.pyplot as plt
from diffusers import UNet1DModel, DDPMScheduler

class ECGDiffusionGenerator:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.scheduler = DDPMScheduler(num_train_timesteps=1000)
        self.unet = UNet1DModel(
            sample_size=384,  # Padded from 360 to be divisible by 16 (4 downsampling blocks)
            in_channels=1,
            out_channels=1,
            down_block_types=("DownBlock1D", "DownBlock1D", "AttnDownBlock1D", "AttnDownBlock1D"),
            up_block_types=("AttnUpBlock1D", "AttnUpBlock1D", "UpBlock1D", "UpBlock1D"),
            block_out_channels=(32, 64, 128, 256)
        ).to(self.device)

    def generate_synthetic_ecg(self, class_label: int, quantity: int, num_inference_steps: int = 50) -> np.ndarray:
        # Start with padded size 384
        noisy_residual = torch.randn((quantity, 1, 384)).to(self.device)
        self.scheduler.set_timesteps(num_inference_steps)
        
        for t in self.scheduler.timesteps:
            with torch.no_grad():
                model_output = self.unet(noisy_residual, t).sample
                noisy_residual = self.scheduler.step(model_output, t, noisy_residual).prev_sample
                
        # Crop back to original sequence length of 360
        return noisy_residual[:, :, :360].cpu().squeeze(1).numpy()

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
