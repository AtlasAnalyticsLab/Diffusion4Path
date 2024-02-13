import torch
import pickle
import torch.nn.functional as F

from tqdm import tqdm
from pathlib import Path

from model.unet import UNet
from scheduler.ddpm import DDPMPipeline
from utils.common import postprocess, create_images_grid
from config.train import training_config
from data.kgh_loader import get_dataloader

def save_images(config, images, i,  folder_name):
    # Postprocess and save sampled images
    images = postprocess(images)
    image_grid = create_images_grid(images, rows=1, cols=1)

    grid_save_dir = Path(config.output_dir, folder_name)
    grid_save_dir.mkdir(parents=True, exist_ok=True)
    image_grid.save(f"{grid_save_dir}/{i:04d}.png")


def evaluate(config, i, pipeline, model, noisy_sample):
    # Perform reverse diffusion process with noisy images.
    noisy_sample = noisy_sample.to(config.device)

    z_values = [torch.randn(noisy_sample.shape).to(config.device) for _ in range(1001)]
    
    images = pipeline.ddpm_beta_sampling(model, noisy_sample, device=config.device, z_values=z_values)
    save_images(config,images,i,"ddpm")

    # # Reverse diffusion for T timesteps
    # images = pipeline.ddim_sampling(model, noisy_sample, device=config.device, z_values=z_values, eta = 0, steps = 500)
    # save_images(config,images,i,"ddim_00")

    # images = pipeline.ddim_sampling(model, noisy_sample, device=config.device, z_values=z_values, eta = 0.5, steps = 500)
    # save_images(config,images,i,"ddim_05")
    
    # images = pipeline.ddim_sampling(model, noisy_sample, device=config.device, z_values=z_values, eta = 1,steps = 500)
    # save_images(config,images,i,"ddim_01")

    # images = pipeline.momentum_sampling(model, noisy_sample, device=config.device, z_values=z_values, eta = 0, steps = 500)
    # save_images(config,images,i,"momentum")

    # images = pipeline.epsilon_sampling(model, noisy_sample, device=config.device, z_values=z_values)
    # save_images(config,images,i,"epsilon")

    # images = pipeline.improved_second_order_sampling(model, noisy_sample, device=config.device, z_values=z_values)
    # save_images(config,images,i,"improved_second")

    # images = pipeline.era_solver_sampling(model, noisy_sample, device=config.device, z_values=z_values)
    # save_images(config,images,i, "era_solver")


def main():
    train_dataloader = get_dataloader(training_config)

    model = UNet(image_size=training_config.image_size,
                 input_channels=training_config.image_channels).to(training_config.device)

    if training_config.use_checkpoint == True:
        checkpoint_path = Path(training_config.checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cuda'))
        model.load_state_dict(checkpoint['model'])
    
    print("Model size: ", sum([p.numel() for p in model.parameters() if p.requires_grad]))

    optimizer = torch.optim.Adam(model.parameters(), lr=training_config.learning_rate)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer,
                                                              T_max=len(train_dataloader) * training_config.num_epochs,
                                                              last_epoch=-1,
                                                              eta_min=1e-9)

    if training_config.resume:
        checkpoint = torch.load(training_config.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        training_config.start_epoch = checkpoint['epoch'] + 1

    for param_group in optimizer.param_groups:
        param_group['lr'] = training_config.learning_rate

    diffusion_pipeline = DDPMPipeline(beta_start=1e-4, beta_end=1e-2, num_timesteps=training_config.diffusion_timesteps)

    global_step = training_config.start_epoch * len(train_dataloader)

    # Training loop
    for epoch in range(training_config.start_epoch, training_config.num_epochs):
        progress_bar = tqdm(total=len(train_dataloader))
        progress_bar.set_description(f"Epoch {epoch}")

        mean_loss = 0

        model.train()
        for step, batch in enumerate(train_dataloader):
            original_images = batch[0].to(training_config.device)
            # print(original_images.shape)
            batch_size = original_images.shape[0]

            # Sample a random timestep for each image
            timesteps = torch.randint(0, diffusion_pipeline.num_timesteps, (batch_size,),
                                      device=training_config.device).long()

            # Apply forward diffusion process at the given timestep
            gamma = None
            noisy_images, noise = diffusion_pipeline.forward_diffusion(original_images, timesteps, gamma)
            noisy_images = noisy_images.to(training_config.device)

            # Predict the noise residual
            noise_pred = model(noisy_images, timesteps)
            loss = F.mse_loss(noise_pred, noise)
            # Calculate new mean on the run without accumulating all the values
            mean_loss = mean_loss + (loss.detach().item() - mean_loss) / (step + 1)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            progress_bar.update(1)
            logs = {"loss": mean_loss, "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
            progress_bar.set_postfix(**logs)
            global_step += 1

        # Evaluation
        if (epoch + 1) % training_config.save_image_epochs == 0 or epoch == training_config.num_epochs - 1:
            model.eval()

            noisy_sample = torch.randn(
                training_config.eval_batch_size,
                training_config.image_channels,
                training_config.image_size,
                training_config.image_size).to(training_config.device)
            
            evaluate(training_config, epoch, diffusion_pipeline, model, noisy_sample)

        if (epoch + 1) % training_config.save_model_epochs == 0 or epoch == training_config.num_epochs - 1:
            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'parameters': training_config,
                'epoch': epoch
            }
            torch.save(checkpoint, Path(training_config.output_dir,
                                        f"unet{training_config.image_size}_e{epoch}.pth"))


if __name__ == "__main__":
    # main()
    no_of_samples = 50
    diffusion_pipeline = DDPMPipeline(beta_start=1e-4, beta_end=1e-2, num_timesteps=training_config.diffusion_timesteps)
    
    # checkpoint_path = Path('/home/de_thak/git/diffusion-ddpm/models/kgh_ip/unet128_e149.pth')
    # checkpoint = torch.load(checkpoint_path, map_location=torch.device('cuda'))

    # main()

    if training_config.generate_samples ==True:
        checkpoint_path = Path(training_config.checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cuda'))

        model =  UNet(image_size=training_config.image_size,
                 input_channels=training_config.image_channels).to(training_config.device)
        model.load_state_dict(checkpoint['model'])

        with open('/home/de_thak/git/diffusion-ddpm/src/noise_values.pkl', 'rb') as file:
            x_T_values = pickle.load(file)
        total_x_T_values = len(x_T_values)


        for i in range(1,no_of_samples):
            noisy_sample = x_T_values[i%total_x_T_values].to(training_config.device)
            evaluate(training_config, i, diffusion_pipeline, model, noisy_sample)

    # model =  UNet(image_size=training_config.image_size,
    #              input_channels=training_config.image_channels).to(training_config.device)
    # model.load_state_dict(checkpoint['model'])

    # with open('/home/de_thak/git/diffusion-ddpm/src/noise_values.pkl', 'rb') as file:
    #     x_T_values = pickle.load(file)
    # total_x_T_values = len(x_T_values)

    # for i in range(1,no_of_samples):
    #     noisy_sample = x_T_values[i%total_x_T_values].to(training_config.device)
    #     evaluate(training_config, i, diffusion_pipeline, model, noisy_sample)
