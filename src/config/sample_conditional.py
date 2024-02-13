from dataclasses import dataclass
import datetime

# Get the current date and time
current_time = datetime.datetime.now()

# Format the date and time as dd_mm_hh_min
formatted_time = current_time.strftime("%d_%m_%H_%M_2")

@dataclass
class SampleConfig:
    image_size = 128
    image_channels = 3
    train_batch_size = 8
    eval_batch_size = 1
    sample_batch_size = 64
    num_eval_samples = 5
    num_epochs = 130
    start_epoch = 88
    learning_rate = 2e-5
    diffusion_timesteps = 1000
    save_image_epochs = 1
    save_model_epochs = 5
    num_classes = 5
    num_samples=2000
    drop_prob = 0.1
    image_dir="/home/de_thak/master_project/dataset/PKGH_224/"
    dataset = 'PKGH_224'
    training_algo = "ddpm"
    gamma = 0.1
    # output_dir = f'/home/de_thak/git/diffusion-ddpm/1601-0533/'
    output_dir = f'/home/de_thak/git/diffusion-ddpm/final_experiments/results_conditional/PKGH_224/{training_algo}/{formatted_time}/'
    samples_dir = f'/home/de_thak/git/diffusion-ddpm/results_conditional_samples/PKGH_224/{training_algo}/'
    generate_samples = True
    checkpoint_path = f'/home/de_thak/git/diffusion-ddpm/results_conditional/PKGH_224/{training_algo}/02_02_15_56/unet128_e49.pth'
    # checkpoint_path = f'/home/de_thak/git/diffusion-ddpm/results_conditional/PKGH_224/{training_algo}/02_02_20_54/unet128_e49.pth'
    sampling_algorithms = ["ddpm","epsilon","improved_second_order"]
    # sampling_algorithms = ["ddim","momentum"]
    # sampling_algorithms = ["ddpm","epsilon","improved_second_order","ddim","momentum"]
    noise_file = f"batch_{sample_batch_size}_1"
    use_checkpoint = True
    conditional = True
    device = "cuda"
    seed = 0
    resume = None

    def __init__(self):
        print("Time", datetime.datetime.now())
        print("Num Samples:", self.num_samples)
        print("Training Algorithm:", self.training_algo)
        print("Gamma(if algorithm is ddpm_ip, else None):", self.gamma)
        print("Output Directory:", self.output_dir)
        print("Drop Probability:", self.drop_prob)
        print("Image Directory:", self.image_dir)
        print("Generate Samples:", self.generate_samples)
        print("Checkpoint Path:", self.checkpoint_path)
        print("Image Size:", self.image_size)
        print("Image Channels:", self.image_channels)
        print("Train Batch Size:", self.train_batch_size)
        print("Eval Batch Size:", self.eval_batch_size)
        print("Num Eval Samples:", self.num_eval_samples)
        print("Num Epochs:", self.num_epochs)
        print("Start Epoch:", self.start_epoch)
        print("Learning Rate:", self.learning_rate)
        print("Diffusion Timesteps:", self.diffusion_timesteps)
        print("Save Image Epochs:", self.save_image_epochs)
        print("Save Model Epochs:", self.save_model_epochs)
        print("Num Classes:", self.num_classes)
        print("Use Checkpoint:", self.use_checkpoint)
        print("Conditional:", self.conditional)

sampling_conditional_config = SampleConfig()