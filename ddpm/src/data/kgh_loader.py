from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader

def get_dataloader(config):
    # dataset_path = load_dataset(dataset, split="train")

    dataset = ImageFolder(
        root=config.image_dir,
        transform=transforms.Compose(
            [
                transforms.Resize((config.image_size, config.image_size)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
    )
    loader = DataLoader(dataset, batch_size=config.train_batch_size, shuffle=True)
    return loader
