import torch
import numpy as np

from tqdm import tqdm

from utils.common import broadcast


class DDPMPipeline:
    def __init__(self, beta_start=1e-4, beta_end=1e-2, num_timesteps=1000):
        # Betas settings are in section 4 of https://arxiv.org/pdf/2006.11239.pdf
        # Implemented linear schedule for now, cosine works better tho.
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps)
        self.alphas = 1 - self.betas

        # alpha-hat in the paper, precompute them
        self.alphas_hat = torch.cumprod(self.alphas, dim=0)

        self.num_timesteps = num_timesteps
        self.num_classes = 5

    def forward_diffusion(self, images, timesteps, gamma=None) -> tuple[torch.Tensor, torch.Tensor]:
        """
        https://arxiv.org/pdf/2006.11239.pdf, equation (14), the term inside epsilon_theta
        :return:
        """
        gaussian_noise = torch.randn(images.shape).to(images.device)

        if gamma is not None:  # Apply input perturbation during training
            gaussian_noise = gaussian_noise + gamma * torch.randn_like(gaussian_noise)

        t = timesteps.to('cpu')
        alpha_hat = self.alphas_hat[t].to(images.device)
        alpha_hat = broadcast(alpha_hat, images)

        return torch.sqrt(alpha_hat) * images + torch.sqrt(1 - alpha_hat) * gaussian_noise, gaussian_noise

    def reverse_diffusion(self, model, noisy_images, timesteps):
        predicted_noise = model(noisy_images, timesteps)
        return predicted_noise

    @torch.no_grad()
    def ddpm_sampling(self, model, initial_noise, device, z_values,guide_w,labels, save_all_steps=False):
        """
        :param model:
        :param initial_noise:
        :param device:
        :param save_all_steps:
        :return:
        """
        image = initial_noise
        images = []

        # don't drop context at test time
        context_mask = torch.zeros_like(labels).to(device)

        for timestep in tqdm(range(self.num_timesteps - 1, -1, -1)):
            # Broadcast timestep for batch_size
            z = z_values[timestep].to(device)
            ts = timestep * torch.ones(image.shape[0], dtype=torch.long, device=device)
           
            predicted_noise = model(image, ts, None, labels)
            if guide_w > 0:
                    uncond_predicted_noise = model(image, ts, None, None)
                    predicted_noise = ((1+guide_w) * predicted_noise) - (guide_w*uncond_predicted_noise)
            beta_t = self.betas[timestep].to(device)
            alpha_t = self.alphas[timestep].to(device)
            alpha_hat = self.alphas_hat[timestep].to(device)

            # Algorithm 2, step 4: calculate x_{t-1} with alphas and variance.
            # Since paper says we can use fixed variance (section 3.2, in the beginning),
            # we will calculate the one which assumes we have x0 deterministically set to one point.
            alpha_hat_prev = self.alphas_hat[timestep - 1].to(device)
            beta_t_hat = (1 - alpha_hat_prev) / (1 - alpha_hat) * beta_t
            
            variance = torch.sqrt(beta_t_hat) * z if timestep > 0 else 0

            image = torch.pow(alpha_t, -0.5) * (image -
                                                beta_t / torch.sqrt((1 - alpha_hat_prev)) *
                                                predicted_noise) + variance
            if save_all_steps:
                images.append(image.cpu())
        return images if save_all_steps else image

    @torch.no_grad()
    def ddim_sampling(self, model, initial_noise, device, z_values, eta, steps,guide_w,labels, epsilon_scale, improved_gamma, save_all_steps=False):
        """
        Algorithm 2 from the paper https://arxiv.org/pdf/2006.11239.pdf
        Seems like we have two variations of sampling algorithm: iterative and with reparametrization trick (equation 15)
        Iterative assumes you have to denoise image step-by-step on T=1000 timestamps, while the second approach lets us
        calculate x_0 approximation constantly without gradually denosing x_T till x_0.

        :param model:
        :param initial_noise:
        :param device:
        :param save_all_steps:
        :return:
        """
        image = initial_noise
        images = []

        # Define an empty list to store timestep values
        a = self.num_timesteps // steps

        # add one to get the final alpha values right (the ones from first scale to data during sampling)
        time_steps = np.asarray(list(range(0, self.num_timesteps, a)))
        time_steps = time_steps + 1
        # previous sequence
        time_steps_prev = np.concatenate([[0], time_steps[:-1]])


        for timestep in tqdm(reversed(range(0, steps))):
            
            z = z_values[timestep].to(device)

            # ts = torch.full((image.shape[0],), time_steps[timestep], device=device, dtype=torch.long)
            # prev_t = torch.full((image.shape[0],), time_steps_prev[timestep], device=device, dtype=torch.long)
            ts = time_steps[timestep] * torch.ones(image.shape[0], dtype=torch.long, device=device)
            prev_t = time_steps_prev[timestep] * torch.ones(image.shape[0], dtype=torch.long, device=device)

 
            #Improved Second Order sampling with DDIM
            if improved_gamma is not None:
                if time_steps[timestep] < time_steps[steps-1]:
                    predicted_prev_noise = predicted_noise

            #DDIM Sampling
            predicted_noise = model(image, ts, None, labels)
            if guide_w > 0:
                    uncond_predicted_noise = model(image, ts, None, None)
                    predicted_noise = ((1+guide_w) * predicted_noise) - (guide_w*uncond_predicted_noise)
            
            #Epsilon scaling sampling with DDIM
            if epsilon_scale is not None:
                predicted_noise = predicted_noise / epsilon_scale
            
            #Improved Second Order sampling with DDIM
            if improved_gamma is not None:
                if time_steps[timestep] < time_steps[steps-1]:
                    predicted_noise = (improved_gamma * predicted_noise) + (1-improved_gamma) * predicted_prev_noise
            
            
            alpha_t = self.alphas_hat[time_steps[timestep] ].to(device)
            alpha_t_prev = self.alphas_hat[time_steps_prev[timestep]].to(device)

            # alpha_t = alpha_t.unsqueeze(1).unsqueeze(2).unsqueeze(3)
            # alpha_t_prev = alpha_t_prev.unsqueeze(1).unsqueeze(2).unsqueeze(3)
            sigma_t = eta * torch.sqrt((1 - alpha_t_prev) / (1 - alpha_t) * (1 - alpha_t / alpha_t_prev))
            # print(sqrt_alphat_prev_by_alpha_t, sqrt_alphat_prev_by_alpha_t.shape)

            image = (
                        torch.sqrt(alpha_t_prev / alpha_t) * image +
                        (torch.sqrt(1 - alpha_t_prev - sigma_t ** 2) - torch.sqrt(
                            (alpha_t_prev * (1 - alpha_t)) / alpha_t)) * predicted_noise +
                        sigma_t * z
                    )
            
            if save_all_steps:
                images.append(image.cpu())
        return images if save_all_steps else image

    