import torch

import matplotlib.pyplot as plt
import matplotlib.animation as animation

from PIL import Image
from pathlib import Path

def broadcast(values, broadcast_to):
    values = values.flatten()

    while len(values.shape) < len(broadcast_to.shape):
        values = values.unsqueeze(-1)

    return values

def all_postprocess(image):
    image = (image / 2 + 0.5).clamp(0, 1)
    
    # Ensure the tensor is on CPU before applying permute
    image = image.cpu()

    # If the tensor has 3 dimensions, you might want to permute the channels
    if image.dim() == 3:
        image = image.permute(1, 2, 0).numpy()
    else:
        # Handle the case where the tensor has more than 3 dimensions
        raise ValueError("Unexpected number of dimensions in the input tensor.")

    image = (image * 255).round().astype("uint8")
    return image

def postprocess(images):
    images = (images / 2 + 0.5).clamp(0, 1)
    images = images.cpu().permute(0, 2, 3, 1).numpy()
    images = (images * 255).round().astype("uint8")
    return images


def create_images_grid(images, rows, cols):
    images = [Image.fromarray(image) for image in images]
    w, h = images[0].size
    grid = Image.new('RGB', size=(cols * w, rows * h))
    for i, image in enumerate(images):
        grid.paste(image, box=(i % cols * w, i // cols * h))
    return grid


def create_sampling_animation(model, pipeline, config, interval=5, every_nth_image=1, rows=2, cols=3):
    # Sample some images from random noise (this is the backward diffusion process).
    # The default pipeline output type is `List[PIL.Image]`
    noisy_sample = torch.randn(
        config.eval_batch_size,
        config.image_channels,
        config.image_size,
        config.image_size).to(config.device)

    # images is a list of num_timesteps images batches, e.g. List[Tensor(NCHW)]
    images = pipeline.sampling(model, noisy_sample, device=config.device, save_all_steps=True)

    fig = plt.figure()
    ims = []
    for i in range(0, pipeline.num_timesteps, every_nth_image):
        imgs = postprocess(images[i])
        image_grid = create_images_grid(imgs, rows=rows, cols=cols)
        im = plt.imshow(image_grid, animated=True)
        ims.append([im])

    plt.axis('off')
    animate = animation.ArtistAnimation(fig, ims, interval=interval, blit=True, repeat_delay=5000)
    path_to_save_animation = Path(config.output_dir, "samples", "diffusion.gif")
    animate.save(str(path_to_save_animation))

