import torch
import pickle
import torch.nn.functional as F
import dataclasses
import inspect
from tqdm import tqdm
from pathlib import Path
import datetime

from model.unet import UNet
from model.unet_conditional import UNet_conditional
from scheduler.ddpm_conditional import DDPMPipeline
from utils.common import postprocess, create_images_grid
from config.train_conditional import training_conditional_config 
from data.kgh_loader import get_dataloader

# Get the current date and time
current_time = datetime.datetime.now()

training_config = training_conditional_config

def save_images(config, images, step, folder_name):
    """Postprocess and save sampled images."""
    images = postprocess(images)
    image_grid = create_images_grid(images, rows=1, cols=1)
    grid_save_dir = Path(config.output_dir, folder_name)
    grid_save_dir.mkdir(parents=True, exist_ok=True)
    image_grid.save(f"{grid_save_dir}/{step}.png")

def generate_samples(config, epoch, pipeline, model, noisy_sample, n_samples, w=1):
    """Generate conditional or unconditional samples using the diffusion pipeline."""
    noisy_sample = noisy_sample.to(config.device)
    z_values = [torch.randn(noisy_sample.shape).to(config.device) for _ in range(1001)]

    for class_id in range(config.num_classes if config.conditional else 1):
        c_i = torch.tensor([class_id]).to(config.device) if config.conditional else None
        images = pipeline.ddpm_beta_sampling(model, noisy_sample, config.device, z_values, w, c_i)
        folder_name = f"ddpm/{w}_{class_id}" if config.conditional else "ddpm/unconditional"
        save_images(config, images, epoch, folder_name)

def evaluate(config, step, pipeline, model, noisy_sample):
    """Evaluate the model by performing the reverse diffusion process."""
    noisy_sample = noisy_sample.to(config.device)
    z_values = [torch.randn(noisy_sample.shape).to(config.device) for _ in range(1001)]
    images = pipeline.ddpm_beta_sampling(model, noisy_sample, config.device, z_values)
    save_images(config, images, step, "ddpm_hat")

def initialize_model(config):
    """Initialize the model based on the conditional flag in the configuration."""
    if config.conditional:
        return UNet_conditional(image_size=config.image_size,
                                input_channels=config.image_channels,
                                num_classes=config.num_classes).to(config.device)
    return UNet(image_size=config.image_size, input_channels=config.image_channels).to(config.device)

def load_checkpoint_if_needed(model, optimizer, lr_scheduler, config):
    """Load checkpoint if resume flag is set, and update the optimizer and scheduler."""
    if config.resume:
        checkpoint = torch.load(config.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        config.start_epoch = checkpoint['epoch'] + 1

    for param_group in optimizer.param_groups:
        param_group['lr'] = config.learning_rate

def main():
    train_dataloader = get_dataloader(training_config)
    model = initialize_model(training_config)

    if training_config.use_checkpoint:
        checkpoint_path = Path(training_config.checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location='cuda')
        model.load_state_dict(checkpoint['model'])
    
    print(f"Model size: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    optimizer = torch.optim.Adam(model.parameters(), lr=training_config.learning_rate)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=len(train_dataloader) * training_config.num_epochs, eta_min=1e-9)

    load_checkpoint_if_needed(model, optimizer, lr_scheduler, training_config)

    diffusion_pipeline = DDPMPipeline(beta_start=1e-4, beta_end=1e-2, num_timesteps=training_config.diffusion_timesteps)
    global_step = training_config.start_epoch * len(train_dataloader)

    # Training loop
    for epoch in range(training_config.start_epoch, training_config.num_epochs):
        progress_bar = tqdm(total=len(train_dataloader), desc=f"Epoch {epoch}")
        mean_loss = 0

        model.train()
        for step, batch in enumerate(train_dataloader):
            original_images, labels = batch[0].to(training_config.device), batch[1].to(training_config.device)
            labels = labels if torch.rand(1).item() >= 0.1 else None

            timesteps = torch.randint(0, diffusion_pipeline.num_timesteps, (original_images.shape[0],),
                                      device=training_config.device).long()
            noisy_images, noise = diffusion_pipeline.forward_diffusion(original_images, timesteps)
            noise_pred = model(noisy_images, timesteps, None, labels) if training_config.conditional else model(noisy_images, timesteps)

            loss = F.mse_loss(noise_pred, noise)
            mean_loss += (loss.detach().item() - mean_loss) / (step + 1)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            progress_bar.update(1)
            logs = {"loss": mean_loss, "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
            progress_bar.set_postfix(**logs)
            global_step += 1

        # Generate samples and save the model
        if (epoch + 1) % training_config.save_image_epochs == 0 or epoch == training_config.num_epochs - 1:
            model.eval()
            noisy_sample = torch.randn(training_config.eval_batch_size,
                                       training_config.image_channels,
                                       training_config.image_size,
                                       training_config.image_size).to(training_config.device)
            generate_samples(training_config, epoch, diffusion_pipeline, model, noisy_sample, training_config.num_eval_samples)

        if (epoch + 1) % training_config.save_model_epochs == 0 or epoch == training_config.num_epochs - 1:
            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'parameters': training_config,
                'epoch': epoch
            }
            torch.save(checkpoint, Path(training_config.output_dir, f"unet{training_config.image_size}_e{epoch}.pth"))

if __name__ == "__main__":
    main()
