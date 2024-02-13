from dataclasses import dataclass


@dataclass
class TrainingConfig:
    image_size = 128
    image_channels = 3
    train_batch_size = 8
    eval_batch_size = 1
    num_eval_samples = 5
    num_epochs = 130
    start_epoch = 20
    learning_rate = 2e-5
    diffusion_timesteps = 1000
    save_image_epochs = 1
    save_model_epochs = 5
    num_classes = 5
    num_samples=50
    drop_prob = 0.1
    image_dir="/home/de_thak/master_project/dataset/PKGH_224/"
    dataset = 'PKGH_224'
    # output_dir = f'/home/de_thak/git/diffusion-ddpm/1601-0533/'
    output_dir = f'/home/de_thak/git/diffusion-ddpm/results_samples/PKGH_224/ddpm/2901-12:00/'
    generate_samples = True
    checkpoint_path = '/home/de_thak/git/diffusion-ddpm/results/PKGH_224/ddpm/2301-04:00/unet128_e129.pth'
    use_checkpoint = True
    conditional = False
    device = "cuda"
    seed = 0
    resume = None


training_config = TrainingConfig()