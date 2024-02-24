from dataclasses import dataclass
import datetime

# Get the current date and time
current_time = datetime.datetime.now()

# Format the date and time as dd_mm_hh_min
formatted_time = current_time.strftime("%d_%m_%H_%M_2")


@dataclass
class AETrainingConfig:
    image_size = 128
    image_channels = 3
    train_batch_size = 8
    eval_batch_size = 1
    num_eval_samples = 5
    num_epochs = 50
    start_epoch = 8
    learning_rate = 1e-5
    save_image_epochs = 1
    save_model_epochs = 5
    num_classes = 5
    num_samples=50
    drop_prob = 0.1
    image_dir="/home/de_thak/diffusion-project/dataset/PKGH_224"
    dataset = 'PKGH_224'
    # output_dir = f'/home/de_thak/git/diffusion-ddpm/1601-0533/'
    output_dir = f'/home/de_thak/diffusion-comparative/helpers/results_AE_PKGH_224/{formatted_time}/'
    generate_samples = True
    checkpoint_path = '/home/de_thak/diffusion-comparative/helpers/results_AE_PKGH_224/22_02_04_21_2/autoencoder128_e29.pth'
    use_checkpoint = False
    conditional = False
    device = "cuda"
    seed = 0
    resume = None


training_config = AETrainingConfig()