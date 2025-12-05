import torch
from diffusers import DDPMPipeline
from torchvision import datasets, transforms

device = "cuda" if torch.cuda.is_available() else "cpu"

pipe = DDPMPipeline.from_pretrained("google/ddpm-cifar10-32").to(device)
pipe.set_progress_bar_config(disable=True)
unet = pipe.unet.eval()

def score_from_noise_pred(eps_pred, sigma):
    return -(1.0 / sigma) * eps_pred

@torch.no_grad()
def map_sigma_to_t(sigma):
    # sigma_t = sqrt((1 - alpha_cumprod) / alpha_cumprod) for DDPM
    alphas_cumprod = pipe.scheduler.alphas_cumprod.to(device)
    sigmas = torch.sqrt((1 - alphas_cumprod) / alphas_cumprod)
    t = int((sigmas - sigma).abs().argmin().item())
    return t

@torch.no_grad()
def score_fn(x, sigma=0.1):
    t = map_sigma_to_t(sigma)
    t_tensor = torch.full((x.shape[0],), t, device=x.device, dtype=torch.long)
    eps_pred = unet(x, t_tensor).sample
    return score_from_noise_pred(eps_pred, sigma), t

def load_cifar_batch(batch_size=8):
    tfm = transforms.Compose([
        transforms.ToTensor(),               # [0,1]
        transforms.Lambda(lambda z: z*2-1)   # scale to [-1,1]
    ])
    ds = datasets.CIFAR10(root="./data", train=False, download=True, transform=tfm)
    idx = torch.randperm(len(ds))[:batch_size]
    batch = torch.stack([ds[i][0] for i in idx]).to(device)
    return batch

if __name__ == "__main__":
    try:
        x = load_cifar_batch(batch_size=8)
        print("Loaded real CIFAR-10 batch.")
    except Exception as e:
        print("Could not load CIFAR-10, using random noise. Err:", e)
        x = torch.randn(8, 3, 32, 32, device=device).clamp(-1, 1)

    sigma = 0.1
    s, t = score_fn(x, sigma=sigma)
    norms = s.flatten(1).norm(dim=1)
    print(f"score shape: {s.shape}")
    print(f"score norms at sigma={sigma} (t={t}): min {norms.min():.3f} max {norms.max():.3f} mean {norms.mean():.3f}")

    # Noise prediction MSE at the same timestep
    t_tensor = torch.full((x.shape[0],), t, device=device, dtype=torch.long)
    noise = torch.randn_like(x)
    noisy = pipe.scheduler.add_noise(original_samples=x, noise=noise, timesteps=t_tensor)
    with torch.no_grad():
        noise_pred = unet(noisy, t_tensor).sample
    mse = (noise_pred - noise).pow(2).mean().item()
    print(f"noise pred MSE at t={t}: {mse:.4f}")
