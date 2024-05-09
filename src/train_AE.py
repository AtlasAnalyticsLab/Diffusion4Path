import torch
import pickle
import torch.nn.functional as F
import dataclasses
import inspect
import itertools

from tqdm import tqdm
from pathlib import Path

from model.unet import UNet
from model.unet_conditional import UNet_conditional
from scheduler.ddpm_conditional import DDPMPipeline
from utils.common import postprocess, create_images_grid
from config.train_autoencoder import AETrainingConfig 
from data.kgh_loader import get_dataloader
from model.LDM.autoencoder import Autoencoder, Encoder, Decoder
import matplotlib.pyplot as plt
import datetime
import torchvision
  # Get the current date and time
current_time = datetime.datetime.now()
training_config = AETrainingConfig

def plot_images(num_epochs, outputs):
    # image plotter
    for k in range(0, num_epochs, 4):
        plt.figure(figsize=(9, 2))
        plt.gray()
        imgs = outputs[k][1].detach().numpy()
        recon = outputs[k][2].detach().numpy()
        for i, item in enumerate(imgs):
            if i >= 9: break
            plt.subplot(2, 9, 9 + i + 1) # row_length + i + 1
            item = item.reshape(-1, 28, 28)
            plt.imshow(item[0])


def plot_images_and_save(original_images, reconstructed_images, epoch, num_plots=2, save_path='plot.png'):
    """
    Plot original and reconstructed images side by side and save to a file.

    Parameters:
    - original_images: Tensor containing original images.
    - reconstructed_images: Tensor containing reconstructed images.
    - num_plots: Number of plots to display.
    - save_path: File path to save the plot.

    Returns:
    - None
    """
    fig, axes = plt.subplots(num_plots, 2, figsize=(8, 2 * 1))

    for i in range(num_plots):
        original_img = torchvision.utils.make_grid(original_images[i], nrow=8, normalize=True).permute(1, 2, 0).numpy()
        recon_img = torchvision.utils.make_grid(reconstructed_images[i], nrow=8, normalize=True).permute(1, 2, 0).numpy()

        axes[i, 0].imshow(original_img)
        axes[i, 0].set_title("Original")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(recon_img)
        axes[i, 1].set_title("Reconstructed")
        axes[i, 1].axis("off")

    plt.tight_layout()

    image_dir = Path(training_config.output_dir)
    image_dir.mkdir(parents=True, exist_ok=True)
    
    # Save the plot to the specified file path
    plt.savefig(f"{image_dir}/{epoch}.png")
    
    # Close the plot to release resources
    plt.close()



def main():
    train_dataloader = get_dataloader(training_config)

    encoder = Encoder(z_channels=4,
                          in_channels=3,
                          channels=128,
                          channel_multipliers=[2, 2, 4],
                          n_resnet_blocks=2)

    decoder = Decoder(out_channels=3,
                          z_channels=4,
                          channels=128,
                          channel_multipliers=[2, 2, 4],
                          n_resnet_blocks=2)
    model = Autoencoder(encoder=encoder, decoder=decoder,z_channels=4,emb_channels=4)
    
    
    if training_config.use_checkpoint == True:
        checkpoint_path = Path(training_config.checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cuda'))
        model.load_state_dict(checkpoint['model'])
    model.to(device=training_config.device)
    
    print("Model size: ", sum([p.numel() for p in model.parameters() if p.requires_grad]))

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=training_config.learning_rate, weight_decay=1e-5)

    if training_config.resume:
        checkpoint = torch.load(training_config.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        training_config.start_epoch = checkpoint['epoch'] + 1

    for param_group in optimizer.param_groups:
        param_group['lr'] = training_config.learning_rate

    global_step = training_config.start_epoch * len(train_dataloader)

    # Training loop
   
    for epoch in range(training_config.start_epoch, training_config.num_epochs):
        
        original_output = []
        recon_images =[]
        
        progress_bar = tqdm(total=len(train_dataloader))
        progress_bar.set_description(f"Epoch {epoch}")

        loss = 0
        sliced_dataloader = itertools.islice(train_dataloader, 10)
        model.train()
        for step, batch in enumerate(sliced_dataloader):
            original_images = batch[0].to(training_config.device)
            labels = batch[1].to(training_config.device)

            recon = model(original_images)

            loss = criterion(recon, original_images)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            progress_bar.update(1)
            logs = {"loss": loss.item(),"step": global_step}
            progress_bar.set_postfix(**logs)
            global_step += 1
            # outputs.append((epoch,original_images,recon))
            # Example usage inside your training loop

            # Save original and reconstructed images for plotting
        
            original_output.append(original_images.cpu())
            recon_images.append(recon.cpu())
        # print("shape", len(original_output), len(recon_images))

        # print(outputs.shape, outputs[0].shape, outputs[1].shape)
         # Evaluation
        if (epoch + 1) % training_config.save_image_epochs == 0 or epoch == training_config.num_epochs - 1:
            model.eval()
            # evaluate(training_config, epoch, diffusion_pipeline, model, noisy_sample)
            # Plot the images after the training loop and save to a file
            plot_images_and_save(original_images=original_output,reconstructed_images=recon_images,epoch=epoch)


        if (epoch + 1) % training_config.save_model_epochs == 0 or epoch == training_config.num_epochs - 1:
            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'parameters': training_config,
                'epoch': epoch
            }

            torch.save(checkpoint, Path(training_config.output_dir,
                                        f"autoencoder{training_config.image_size}_e{epoch}.pth"))


if __name__ == "__main__":
    # main()
    main()

 