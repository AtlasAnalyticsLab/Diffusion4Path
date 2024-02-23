import torch
import pickle
import torch.nn.functional as F
import time
from tqdm import tqdm
from pathlib import Path
import argparse

from model.unet import UNet
from model.unet_conditional import UNet_conditional
from scheduler.ddpm_conditional import DDPMPipeline
from utils.common import postprocess, create_images_grid, all_postprocess
from config.sample_conditional import sampling_conditional_config
from data.kgh_loader import get_dataloader
import datetime
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
import torch.multiprocessing as mp
from PIL import Image
import random
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

gamma_range = (0.1, 1.5)  # Define start and end for gamma
scale_range = (1.01, 1.03)  # Define start and end for scale

# Generate 5 random gamma values
random_gamma_values = [random.uniform(gamma_range[0], gamma_range[1]) for _ in range(5)]

# Generate 5 random scale values
random_scale_values = [random.uniform(scale_range[0], scale_range[1]) for _ in range(10)]
scale_values = [1.014, 1.018] 

# Initialize distributed training
def initialize_dist():
    torch.distributed.init_process_group(backend='nccl')

  # Get the current date and time
current_time = datetime.datetime.now()

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

def save_images(config, images, i,  folder_name):
    # Postprocess and save sampled images
    images = postprocess(images)
    image_grid = create_images_grid(images, rows=1, cols=1)

    # # Format the date and time as dd_mm_hh_min
    # formatted_time = current_time.strftime("%d_%m_%H_%M")
    # folder_name = formatted_time + "/" + folder_name
    # print("folder", folder_name)

    grid_save_dir = Path(config.output_dir, folder_name)
    # grid_save_dir = Path(config.output_dir)
    grid_save_dir.mkdir(parents=True, exist_ok=True)
    image_grid.save(f"{grid_save_dir}/{i}.png")

def perform_computations(config, epoch, pipeline, model, noisy_sample, n_samples, w, z_values, class_id):
    c_i = torch.tensor([class_id]).to(config.device)
    images = []
    experiment_count =0
    if "ddpm" in config.sampling_algorithms:
        start = time.time()
        images += pipeline.ddpm_beta_hat_sampling(model, noisy_sample, config.device, z_values, w, c_i)
        folder_name = "ddpm_hat/" + str(w) + "_" + str(class_id)
        end1 = time.time()
        individual_save_images(config, images, str(epoch), folder_name=folder_name)
        end2 = time.time()
        experiment_count +=1
        # print("ddpm Calculated time", end1 - start, end2 - start)

    if "epsilon" in config.sampling_algorithms:
        for scale in scale_values:
            images += pipeline.epsilon_sampling(model, noisy_sample, config.device, z_values, scale, guide_w=w, labels=c_i)
            folder_name = f"{scale}epsilon/" + str(w) + "_" + str(class_id)
            individual_save_images(config, images, str(epoch), folder_name=folder_name)
            experiment_count +=1
    
    if "improved_second_order" in config.sampling_algorithms:
        for gamma in [1.5]:
            images = pipeline.improved_second_order_sampling(model, noisy_sample, config.device, z_values, guide_w=w, labels=c_i, gamma=gamma)
            folder_name = f"{gamma}improved_second_order/" + str(w) + "_" + str(class_id)
            individual_save_images(config, images, str(epoch), folder_name=folder_name)
            experiment_count +=1
    
    if "ddim" in config.sampling_algorithms:
        for eta in [0, 0.5]:
            images = pipeline.ddim_sampling(model, noisy_sample, config.device, z_values, eta=eta, steps=500, guide_w=w, labels=c_i)
            folder_name = f"{eta}ddim/" + str(w) + "_" + str(class_id)
            individual_save_images(config, images, str(epoch), folder_name=folder_name)
            experiment_count +=1
            
    if "momentum" in config.sampling_algorithms:
        images = pipeline.momentum_sampling(model, noisy_sample, config.device, z_values, eta=0, steps=500, guide_w=w, labels=c_i)
        folder_name = f"momentum/" + str(w) + "_" + str(class_id)
        individual_save_images(config, images, str(epoch), folder_name=folder_name)
        experiment_count +=1
    
    return experiment_count

def generate_samples(config, epoch, pipeline, model, noisy_sample, n_samples, w=1):
    noisy_sample = noisy_sample.to(config.device)
    z_values = [torch.randn(noisy_sample.shape).to(config.device) for _ in range(1001)]

    per_class_count = 0
    if config.conditional:
        processes = []    
        for class_id in tqdm(range(config.num_classes)):
            num_of_experiments = perform_computations(config, epoch, pipeline, model, noisy_sample, n_samples, w, z_values, class_id)
            per_class_count += num_of_experiments*config.sample_batch_size
            print(f"Images generated for one experiment : {per_class_count/num_of_experiments}")
            print(f"Images generated for {class_id+1}/{sampling_conditional_config.num_classes} for experiments {num_of_experiments} : {per_class_count}")
            # p = mp.Process(target=perform_computations, args=(config, epoch, pipeline, model, noisy_sample, n_samples, w, z_values, class_id))
            # processes.append(p)
            # p.start()

        # for p in processes:
        #     p.join()
            
    return per_class_count


def evaluate(config, i, pipeline, model, noisy_sample):
    # Perform reverse diffusion process with noisy images.
    noisy_sample = noisy_sample.to(config.device)

    z_values = [torch.randn(noisy_sample.shape).to(config.device) for _ in range(1001)]
    
    images = pipeline.ddpm_beta_hat_sampling(model, noisy_sample, device=config.device, z_values=z_values)
    save_images(config,images,i,"ddpm_hat")

def parse_args():
    parser = argparse.ArgumentParser(description="SampleConfig Script")
    parser.add_argument("--sample_batch_size", type=int, default=128, help="Sample Batch Size")
    parser.add_argument("--noise_file_name", type=str, default="noise_values_64_1", help="Noise file name")
    parser.add_argument("--sampling_algorithms", nargs='+', default=["ddpm", "epsilon", "improved_second_order"], help="List of sampling algorithms")
    parser.add_argument("--n_samples", type=int, default=2000, help="No of Samples")
     
     # Add more arguments for other parameters as needed

    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()
   
    sampling_conditional_config.sample_batch_size= args.sample_batch_size
    sampling_conditional_config.noise_file = args.noise_file_name
    sampling_conditional_config.sampling_algorithms=args.sampling_algorithms
    sampling_conditional_config.num_samples = args.n_samples

    print("Main Sampling Algorithm:", sampling_conditional_config.sampling_algorithms)
    print("Noise File:", sampling_conditional_config.noise_file)
    print("Sample Batch Size", sampling_conditional_config.sample_batch_size)
    print("Sample Batch Size", sampling_conditional_config.num_samples)
    print("Checkpoint", sampling_conditional_config.checkpoint_path)
    print("Output directory", sampling_conditional_config.output_dir)
    
    no_of_samples = sampling_conditional_config.num_samples
    diffusion_pipeline = DDPMPipeline(beta_start=1e-4, beta_end=1e-2, num_timesteps=sampling_conditional_config.diffusion_timesteps)

    guide_ws = [0,2]
    
    if sampling_conditional_config.generate_samples ==True:
        checkpoint_path = Path(sampling_conditional_config.checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cuda'))

        model =  UNet_conditional(image_size=sampling_conditional_config.image_size,
                 input_channels=sampling_conditional_config.image_channels,num_classes=sampling_conditional_config.num_classes).to(sampling_conditional_config.device)
        model.load_state_dict(checkpoint['model'])

        noise_file = sampling_conditional_config.noise_file
        with open(f'/home/de_thak/diffusion-comparative/helpers/noise_values/noise_values_{noise_file}.pkl', 'rb') as file:
            x_T_values = pickle.load(file)
        total_x_T_values = len(x_T_values)
        
        total_count = 0
        num_samples =(no_of_samples//sampling_conditional_config.sample_batch_size) + 1
        num_samples = (num_samples//sampling_conditional_config.num_classes)+1
        print("num", num_samples)
        for i in tqdm(range(1,num_samples+1)):
            noisy_sample = x_T_values[i%total_x_T_values].to(sampling_conditional_config.device)
            for w in tqdm(guide_ws):
                per_class_count = generate_samples(sampling_conditional_config, i, diffusion_pipeline, model, noisy_sample, int(sampling_conditional_config.num_samples/5), w)
                total_count += per_class_count
                print(f"Total images generated for guidance {len(guide_ws)} and n_class {sampling_conditional_config.num_classes} => {total_count}")
                