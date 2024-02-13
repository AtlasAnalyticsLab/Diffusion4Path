import torch
import pickle
import torch.nn.functional as F
import dataclasses
import inspect

from tqdm import tqdm
from pathlib import Path

from model.unet import UNet
from model.unet_conditional import UNet_conditional
from scheduler.ddpm_conditional import DDPMPipeline
from utils.common import postprocess, create_images_grid
from config.train_conditional import training_conditional_config 
from data.kgh_loader import get_dataloader
import datetime
  # Get the current date and time
current_time = datetime.datetime.now()
training_config = training_conditional_config

def save_images(config, images, i,  folder_name):
    # Postprocess and save sampled images
    images = postprocess(images)
    image_grid = create_images_grid(images, rows=1, cols=1)

    grid_save_dir = Path(config.output_dir, folder_name)
    grid_save_dir.mkdir(parents=True, exist_ok=True)
    image_grid.save(f"{grid_save_dir}/{i}.png")

def generate_samples(config, epoch, pipeline, model, noisy_sample, n_samples,w=1):
    noisy_sample = noisy_sample.to(config.device)
    z_values = [torch.randn(noisy_sample.shape).to(config.device) for _ in range(1001)]
    
    if config.conditional == True:
            for class_id in range(0,config.num_classes):
                    c_i = torch.tensor([class_id]).to(config.device)
                    # def ddpm_beta_sampling(self, model, initial_noise, device, z_values, guide_w, c_i, save_all_steps=False):
                    # print("hi")
                    images = pipeline.ddpm_beta_hat_sampling(model, noisy_sample, config.device, z_values, w, c_i)
                    folder_name = "ddpm_hat/"+str(w)+"_"+str(class_id)
                    
                    save_images(config,images,str(epoch),folder_name=folder_name)

def evaluate(config, i, pipeline, model, noisy_sample):
    # Perform reverse diffusion process with noisy images.
    noisy_sample = noisy_sample.to(config.device)

    z_values = [torch.randn(noisy_sample.shape).to(config.device) for _ in range(1001)]
    
    images = pipeline.ddpm_beta_hat_sampling(model, noisy_sample, device=config.device, z_values=z_values)
    save_images(config,images,i,"ddpm_hat")

def main():
    train_dataloader = get_dataloader(training_config)

    if training_config.conditional == True:
        model = UNet_conditional(image_size=training_config.image_size,
                 input_channels=training_config.image_channels,num_classes=training_config.num_classes).to(training_config.device)
    else :
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
            labels = batch[1].to(training_config.device)

            # Randomly set labels to None 10% of the time
            if torch.rand(1).item() < 0.1:
                labels = None
            
            # print(labels)
            
            batch_size = original_images.shape[0]

            # Sample a random timestep for each image
            timesteps = torch.randint(0, diffusion_pipeline.num_timesteps, (batch_size,),
                                      device=training_config.device).long()

            # Apply forward diffusion process at the given timestep
            if training_config.training_algo == "ddpm_ip":
                gamma = training_config.gamma
            else:
                gamma = None
                
            noisy_images, noise = diffusion_pipeline.forward_diffusion(original_images, timesteps, gamma)
            noisy_images = noisy_images.to(training_config.device)

            # Predict the noise residual
            if training_config.conditional == True:
                noise_pred = model(noisy_images, timesteps, None, labels)
            else:
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
            
            # evaluate(training_config, epoch, diffusion_pipeline, model, noisy_sample)
            generate_samples(training_config, epoch, diffusion_pipeline, model, noisy_sample, training_config.num_eval_samples)

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
    # Display GPU memory summary
    torch.cuda.memory_summary(device=None, abbreviated=False)
    main()

 