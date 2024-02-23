from dataclasses import dataclass
import datetime
import inspect
import dataclasses
# Get the current date and time
current_time = datetime.datetime.now()

# Format the date and time as dd_mm_hh_min
formatted_time = current_time.strftime("%d_%m_%H_%M")

@dataclass
class TrainingLDMConfig:
    latent_size = 16
    latent_channels = 4
    image_size = 128
    image_channels = 3
    train_batch_size = 32
    eval_batch_size = 1
    num_eval_samples = 5
    num_epochs = 50
    start_epoch = 0
    learning_rate = 2e-5
    diffusion_timesteps = 1000
    save_image_epochs = 5
    save_model_epochs = 5
    num_classes = 5
    num_samples=50
    drop_prob = 0.1
    image_dir="/home/de_thak/master_project/dataset/PKGH_224/"
    dataset = 'PKGH_224'
    training_algo = "ldm"
    gamma = 0.1
    # output_dir = f'/home/de_thak/git/diffusion-ddpm/1601-0533/'
    output_dir = f'/home/de_thak/diffusion-comparative/helpers/experiments/results_conditional/PKGH_224/{training_algo}/{formatted_time}/'
    # /home/de_thak/diffusion-comparative/helpers/experiments/results_conditional/PKGH_224/{training_algo}/{formatted_time}/
    generate_samples = True
    ae_checkpoint_path = '/home/de_thak/diffusion-comparative/helpers/results_AE_PKGH_224/22_02_04_21_2/autoencoder128_e29.pth'
    checkpoint_path = '/home/de_thak/git/diffusion-ddpm/results_conditional/PKGH_224/ddpm/2701-10:50/unet128_e84.pth'
    use_checkpoint = False
    conditional = True
    device = "cuda"
    seed = 0
    resume = None

    def __init__(self):
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
        print("Num Samples:", self.num_samples)
        print("Drop Probability:", self.drop_prob)
        print("Image Directory:", self.image_dir)
        print("Training Algorithm:", self.training_algo)
        print("Gamma(if algorithm is ddpm_ip, else None):", self.gamma)
        print("Output Directory:", self.output_dir)
        print("Generate Samples:", self.generate_samples)
        print("Checkpoint Path:", self.checkpoint_path)
        print("Use Checkpoint:", self.use_checkpoint)
        print("Conditional:", self.conditional)


training_conditional_config = TrainingLDMConfig()
