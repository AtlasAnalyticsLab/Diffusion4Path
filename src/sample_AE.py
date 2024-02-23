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
from config.autoencoder import AETrainingConfig 
from data.kgh_loader import get_dataloader
from model.LDM.autoencoder import Autoencoder, Encoder, Decoder
import matplotlib.pyplot as plt
import datetime
import torchvision
from utils.common import postprocess, create_images_grid, all_postprocess
from PIL import Image
  # Get the current date and time
current_time = datetime.datetime.now()
training_config = AETrainingConfig


def individual_save_images(config, images, i, folder_name):
    # Ensure the output directory exists
    save_dir = Path(config.output_dir, folder_name)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Iterate through the list of images and save each one individually
    for idx, image in enumerate(images):
        # Postprocess and save each sampled image individually
        postprocessed_image = all_postprocess(image)
        image_path = f"{save_dir}/{i}_{idx}.png"
        Image.fromarray(postprocessed_image).save(image_path)

def plot_images_and_save(original_images, reconstructed_images, epoch, num_plots=1, save_path='plot.png'):
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
    original_images.cpu()
    reconstructed_images.cpu()
    # for i in range(num_plots):
    original_img = torchvision.utils.make_grid(original_images.cpu(), nrow=4, normalize=True).permute(1, 2, 0).numpy()
    recon_img = torchvision.utils.make_grid(reconstructed_images.cpu(), nrow=4, normalize=True).permute(1, 2, 0).numpy()

    axes[0].imshow(original_img)
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(recon_img)
    axes[1].set_title("Reconstructed")
    axes[1].axis("off")


    save_dir = Path(training_config.output_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
  
    plt.tight_layout()

    image_dir = Path(training_config.output_dir)
    image_dir.mkdir(parents=True, exist_ok=True)
    
    # Save the plot to the specified file path
    plt.savefig(f"{image_dir}/{epoch}.png")
    
    # Close the plot to release resources
    plt.close()
  
    # save_dir = Path(training_config.output_dir, 'original')
    # save_dir.mkdir(parents=True, exist_ok=True)

    # for idx, image in enumerate(original_images):
    #     # Postprocess and save each sampled image individually
    #     postprocessed_image = all_postprocess(image)
    #     image_path = f"{save_dir}/{epoch}_{idx}.png"
    #     Image.fromarray(postprocessed_image).save(image_path)
        
    # save_dir = Path(training_config.output_dir, 'recon')
    # save_dir.mkdir(parents=True, exist_ok=True)
    
    # for idx, image in enumerate(reconstructed_images):
    #     # Postprocess and save each sampled image individually
    #     postprocessed_image = all_postprocess(image)
    #     image_path = f"{save_dir}/{epoch}_{idx}.png"
    #     Image.fromarray(postprocessed_image).save(image_path)

    



def main():
    train_dataloader = get_dataloader(training_config)

    encoder = Encoder(z_channels=4,
                          in_channels=3,
                          channels=128,
                          channel_multipliers=[1, 2, 4, 4],
                          n_resnet_blocks=2)

    decoder = Decoder(out_channels=3,
                          z_channels=4,
                          channels=128,
                          channel_multipliers=[1, 2, 4, 4],
                          n_resnet_blocks=2)
    model = Autoencoder(encoder=encoder, decoder=decoder,z_channels=4,emb_channels=4)
    
    
    if training_config.use_checkpoint == True:
        checkpoint_path = Path(training_config.checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cuda'))
        model.load_state_dict(checkpoint['model'])
    model.to(device=training_config.device)
    
    print("Model size: ", sum([p.numel() for p in model.parameters() if p.requires_grad]))
 
    original_output = []
    recon_images =[]

    model.eval()
    for step, batch in enumerate(train_dataloader):
            original_images = batch[0].to(training_config.device)
            recon = model(original_images)

            plot_images_and_save(original_images=original_images,reconstructed_images=recon,epoch=step)


if __name__ == "__main__":
    # main()
    main()

 