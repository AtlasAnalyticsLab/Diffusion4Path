from dataclasses import dataclass
import datetime
import inspect
import dataclasses
# Get the current date and time
current_time = datetime.datetime.now()

# Format the date and time as dd_mm_hh_min
formatted_time = current_time.strftime("%d_%m_%H_%M")

@dataclass
class TrainingConfig:
    image_size = 128
    image_channels = 3
    train_batch_size = 8
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
    image_dir=""
    dataset = 'PKGH_224'
    training_algo = "ddpm"
    gamma = 0.1
    output_dir = f''
    generate_samples = True
    checkpoint_path = ''
    use_checkpoint = False
    conditional = True
    device = "cuda"
    seed = 0
    resume = None

training_conditional_config = TrainingConfig()
