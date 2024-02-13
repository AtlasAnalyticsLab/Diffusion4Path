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
    def ddpm_beta_sampling(self, model, initial_noise, device, z_values, save_all_steps=False):
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
        for timestep in tqdm(range(self.num_timesteps - 1, -1, -1)):
            # Broadcast timestep for batch_size
            z = z_values[timestep].to(device)
            ts = timestep * torch.ones(image.shape[0], dtype=torch.long, device=device)
            predicted_noise = model(image, ts)
            predicted_noise = predicted_noise
            # print("scale added")
            beta_t = self.betas[timestep].to(device)
            alpha_t = self.alphas[timestep].to(device)
            alpha_hat = self.alphas_hat[timestep].to(device)

            # Algorithm 2, step 4: calculate x_{t-1} with alphas and variance.
            # Since paper says we can use fixed variance (section 3.2, in the beginning),
            # we will calculate the one which assumes we have x0 deterministically set to one point.
            alpha_hat_prev = self.alphas_hat[timestep - 1].to(device)
            beta_t_hat = (1 - alpha_hat_prev) / (1 - alpha_hat) * beta_t
            variance = torch.sqrt(beta_t) * z if timestep > 0 else 0

            image = torch.pow(alpha_t, -0.5) * (image -
                                                beta_t / torch.sqrt((1 - alpha_hat_prev)) *
                                                predicted_noise) + variance
            if save_all_steps:
                images.append(image.cpu())
        return images if save_all_steps else image

    @torch.no_grad()
    def ddpm_beta_hat_sampling(self, model, initial_noise, device, z_values, save_all_steps=False):
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
        for timestep in tqdm(range(self.num_timesteps - 1, -1, -1)):
            # Broadcast timestep for batch_size
            z = z_values[timestep].to(device)
            ts = timestep * torch.ones(image.shape[0], dtype=torch.long, device=device)
            predicted_noise = model(image, ts)
            predicted_noise = predicted_noise
            # print("scale added")
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
    def ddim_sampling(self, model, initial_noise, device, z_values, eta, steps, save_all_steps=False):
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

            t = torch.full((image.shape[0],), time_steps[timestep], device=device, dtype=torch.long)
            prev_t = torch.full((image.shape[0],), time_steps_prev[timestep], device=device, dtype=torch.long)

            epsilon_theta_t = model(image, t)

            # get current and previous alpha_cumprod
            # t.to('cpu')
            # prev_t.to('cpu')
            t_index = t.to('cpu')
            t_prev_index = prev_t.to('cpu')
            alpha_t = self.alphas_hat[t_index].to(device)
            alpha_t_prev = self.alphas_hat[t_prev_index].to(device)

            sigma_t = eta * torch.sqrt((1 - alpha_t_prev) / (1 - alpha_t) * (1 - alpha_t / alpha_t_prev))
            epsilon_t = z_values[timestep].to(device)

            image = (
                torch.sqrt(alpha_t_prev / alpha_t) * image +
                (torch.sqrt(1 - alpha_t_prev - sigma_t ** 2) - torch.sqrt(
                    (alpha_t_prev * (1 - alpha_t)) / alpha_t)) * epsilon_theta_t +
                sigma_t * epsilon_t
            )

            if save_all_steps:
                images.append(image.cpu())
        return images if save_all_steps else image

    @torch.no_grad()
    def epsilon_sampling(self, model, initial_noise, device, z_values, save_all_steps=False):
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
        for timestep in tqdm(range(self.num_timesteps - 1, -1, -1)):
            # Broadcast timestep for batch_size
            z = z_values[timestep].to(device)
            ts = timestep * torch.ones(image.shape[0], dtype=torch.long, device=device)
            predicted_noise = model(image, ts)
            predicted_noise = predicted_noise / 1.005
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
    def improved_second_order_sampling(self, model, initial_noise, device, z_values, save_all_steps=False):
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
        for timestep in tqdm(range(self.num_timesteps - 1, -1, -1)):
            # Broadcast timestep for batch_size
            z = z_values[timestep].to(device)
            ts = timestep * torch.ones(image.shape[0], dtype=torch.long, device=device)
            ts_plus_1 = (timestep+1) * torch.ones(image.shape[0], dtype=torch.long, device=device)

            predicted_noise = model(image, ts)
            if timestep < self.num_timesteps - 1:
                prev_predicted_noise = model(image, ts_plus_1)
                predicted_noise = (2 * predicted_noise) - prev_predicted_noise
            
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
    def adaptive_momentum_sampling(self, model, initial_noise, device, z_values, eta, steps, save_all_steps=False):
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
        time_steps_prev = np.concatenate([[0], time_steps[:-1]])

        print("timestep", time_steps, time_steps_prev)

        b = torch.tensor(0.15, device=device)
        a = torch.tensor((1-0.15), device=device)
        # a = torch.sqrt(1 - b ** 2)


        c = torch.tensor(0.001, device=device)

        m_t = torch.tensor(0.0, device=device)
        v_t = torch.tensor(1.0, device=device)

        zeta = torch.tensor(1e-8, device=device)

        for timestep in tqdm(reversed(range(0, steps))):
         
            t = torch.full((image.shape[0],), time_steps[timestep], device=device, dtype=torch.long)
            prev_t = torch.full((image.shape[0],), time_steps_prev[timestep], device=device, dtype=torch.long)

            # get current and previous alpha_cumprod
            # t.to('cpu')
            # prev_t.to('cpu')
            t_index = t.to('cpu')
            t_prev_index = prev_t.to('cpu')
            alpha_t = self.alphas_hat[t_index].to(device)
            alpha_t_prev = self.alphas_hat[t_prev_index].to(device)

            epsilon_theta_t = model(image, t)

            sigma_t = eta * torch.sqrt((1 - alpha_t_prev) / (1 - alpha_t) * (1 - alpha_t / alpha_t_prev))
            epsilon_t = z_values[timestep].to(device)
            
            mean = torch.sqrt((1 - alpha_t_prev - (sigma_t ** 2)) / alpha_t_prev) -  torch.sqrt((1 - alpha_t) / alpha_t)
            xt = image / torch.sqrt(alpha_t)

            d_bar_xt = (mean * epsilon_theta_t) + ((sigma_t / alpha_t_prev) * epsilon_t)
            image = ((image/torch.sqrt(alpha_t)) + (epsilon_theta_t * mean) + ((sigma_t * epsilon_t)/alpha_t_prev))

            # v_t = (1 - c) * v_t + c * torch.norm(d_bar_xt)**2
            # m_t = a * m_t + b * d_bar_xt

            # image = alpha_t_prev * (xt  + (m_t / (torch.sqrt(v_t) * zeta)))

            if save_all_steps:
                images.append(image.cpu())
        return images if save_all_steps else image

    @torch.no_grad()
    def momentum_sampling(self, model, initial_noise, device, z_values, eta, steps, save_all_steps=False):
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

        b = torch.tensor(0.5, device=device)
        a = torch.tensor(0.5, device=device)
        # a = torch.sqrt(1 - b ** 2)


        c = torch.tensor(0.001, device=device)

        m_t = torch.tensor(0.0, device=device)
        v_t = torch.tensor(1.0, device=device)

        zeta = torch.tensor(1e-8, device=device)


        for timestep in tqdm(reversed(range(0, steps))):
            
            z = z_values[timestep].to(device)

            t = torch.full((image.shape[0],), time_steps[timestep], device=device, dtype=torch.long)
            prev_t = torch.full((image.shape[0],), time_steps_prev[timestep], device=device, dtype=torch.long)

            epsilon_theta_t = model(image, t)

            # get current and previous alpha_cumprod
            # t.to('cpu')
            # prev_t.to('cpu')
            t_index = t.to('cpu')
            t_prev_index = prev_t.to('cpu')
            alpha_t = self.alphas_hat[t_index].to(device)
            alpha_t_prev = self.alphas_hat[t_prev_index].to(device)

            sigma_t = eta * torch.sqrt((1 - alpha_t_prev) / (1 - alpha_t) * (1 - alpha_t / alpha_t_prev))
            epsilon_t = z_values[timestep].to(device)

            xt_bar =  image / torch.sqrt(alpha_t)  
            mean = ((torch.sqrt(1 - alpha_t_prev - sigma_t ** 2) - torch.sqrt(
                    (alpha_t_prev * (1 - alpha_t)) / alpha_t)) ) / torch.sqrt(alpha_t_prev)
            variance = sigma_t / torch.sqrt(alpha_t_prev)

            dxt_bar =  mean * epsilon_theta_t + variance * epsilon_t 

            squared_norm_dxt_bar = torch.sum(dxt_bar**2)
            
           
            # image = alpha_t_prev * (xt_bar  + (m_t))

            v_t = ((1 - c) * v_t) + (c * squared_norm_dxt_bar)
            m_t = (a * m_t) + (b * dxt_bar)



            image = torch.sqrt(alpha_t_prev) * (xt_bar + m_t)
            

            if save_all_steps:
                images.append(image.cpu())
        return images if save_all_steps else image
    
    @torch.no_grad()
    def era_solver_sampling(self, model, initial_noise, device, z_values, save_all_steps=False):
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

        # Define helper functions for Lagrange interpolation and error-robust selection
        image = initial_noise
        images = []

        def lagrange_interpolation(t, tau, m,k, buffer):
            result = 1.0
            for l in range(0,k-1):
                if l != m:
                    if tau[l] != tau[m]:
                        # print("lagrange_interpolation", (t - buffer[tau[l]][0]) / buffer[tau[m]][0] - buffer[tau[l]][0]) 
                        # print("start result", result)
                        result *= (t - buffer[tau[l]][0]) / (buffer[tau[m]][0] - buffer[tau[l]][0])
                        
                        # print("t",t)
                        # print("l",l, tau[l], buffer[tau[l]][0])
                        # print("m",m,tau[m], buffer[tau[m]][0])
                        # print("nom", (buffer[tau[m]][0] - buffer[tau[l]][0]))
                        # print("denom", (buffer[tau[m]][0] - buffer[tau[l]][0]))
                        # print("lagrange_interpolation result", result)
            return result

        def lagrange_function(i,t, k, tau, buffer):
            k = len(tau)
            # print("buffer length", len(buffer), k, len(tau))
            result = 0.0
            
            for m in range(0,1):
                # print("hi", t,m,tau[m],  buffer[tau[m]], tau, tau_bar)
                epsilon_theta = buffer[tau[m]]
                
                lm = lagrange_interpolation(t, tau, m, k, buffer)
                result += lm * epsilon_theta[-1]
                # print("lagrange_function",m, result, lm,  epsilon_theta[-1])
 
                # print("lagrange epsilon result", lm, epsilon_theta[-1])
            return result

        def calculate_error_measure(estimated_noise, predicted_noise):
            """
            Calculate the error measure for estimated noise.
            """
            # print("estimated", estimated_noise)
            # print("predicted", predicted_noise)
            return torch.norm(estimated_noise - predicted_noise, p=2)
        
        def calculate_tau_bar(i, k, buffer):
            """
            Calculate {τ¯m} based on Eq. 16
            """
           
            # print("tau bar", i, k , buffer)
            tau_bar = torch.zeros(k, dtype=torch.long)
            
            for m in range(1,k):
                # print("tau_bar i",i, k, m)
                # print("m", m, k-1)
                tau_bar[m] = (i/k) * m
            # print("tau bar", tau_bar)
            return tau_bar

        def calculate_tau( i, tau_bar, k, delta_epsilon, lambda_val):
            """
            Calculate {τm} based on Eq. 17 
            """
            tau = torch.zeros(k, dtype=torch.long)
            
            for m in range(1,k):
                tau[m] = torch.pow((tau_bar[m]/i), delta_epsilon/lambda_val) * i
            # print("taur i",tau, delta_epsilon)
            return tau
        
        k = 100
        # Initialize the buffer
        buffer = []
        lambda_val = 0.5
        delta_epsilon = lambda_val
        eta=0
        steps = 500
        new=0

        # Define an empty list to store timestep values
        a = self.num_timesteps // steps

        # add one to get the final alpha values right (the ones from first scale to data during sampling)
        time_steps = np.asarray(list(range(0, self.num_timesteps, a)))
        time_steps = time_steps + 1
        # previous sequence
        time_steps_prev = np.concatenate([[0], time_steps[:-1]])
        count = 0
        # ERA algorithm modifications
        for timestep in tqdm(reversed(range(0, steps))):
            # Broadcast timestep for batch_size
            z = z_values[timestep].to(device)

            t = torch.full((image.shape[0],), time_steps[timestep], device=device, dtype=torch.long)
            prev_t = torch.full((image.shape[0],), time_steps_prev[timestep], device=device, dtype=torch.long)
            
            epsilon_theta_t = model(image, t)

            # get current and previous alpha_cumprod
            # t.to('cpu')
            # prev_t.to('cpu')
            t_index = t.to('cpu')
            t_prev_index = prev_t.to('cpu')
            alpha_t = self.alphas_hat[t_index].to(device)
            alpha_t_prev = self.alphas_hat[t_prev_index].to(device)

            j=-1
           
            if timestep == steps-1:
                print("hi hello")
                lambda_val = 0.5
                delta_epsilon = lambda_val
                buffer.append((t, epsilon_theta_t))

            else:
                
                if ((steps-1) - timestep) < k:
                    # print("hi", steps-1, timestep, k-1)
                    sigma_t = eta * torch.sqrt((1 - alpha_t_prev) / (1 - alpha_t) * (1 - alpha_t / alpha_t_prev))
                    z = z_values[timestep].to(device)
                    image = (
                        torch.sqrt(alpha_t_prev / alpha_t) * image +
                        (torch.sqrt(1 - alpha_t_prev - sigma_t ** 2) - torch.sqrt(
                            (alpha_t_prev * (1 - alpha_t)) / alpha_t)) * epsilon_theta_t +
                        sigma_t * z
                    )
                   
                    t_minus_1 = torch.full((image.shape[0],), time_steps[timestep-1], device=device, dtype=torch.long)
                    epsilon_theta_t_minus_1 = model(image, t_minus_1)

                    # print("tell me", epsilon_theta_t_minus_1)

                    buffer.append((t_minus_1, epsilon_theta_t_minus_1))
                    new+=1
                else : 
                    count+=1
                    tau_bar = calculate_tau_bar( steps-timestep, k, buffer)
                    tau = calculate_tau( steps-timestep,tau_bar, k, delta_epsilon, lambda_val)
                    
                    # print("timestep  buffer", time_steps[timestep], buffer[j-1][0],  buffer[j-2][0])
                    epsilon_t_minus_1_bar = lagrange_function( steps-timestep,time_steps[timestep] , k, tau, buffer)
                    # print("epsilon",epsilon_t_minus_1_bar )
                    
                    epsilon_theta_t_plus_1 = buffer[j-1][1]
                    epsilon_theta_t_plus_2 = buffer[j-2][1]
                    epsilon_t = (0.0416) * ((9 * epsilon_t_minus_1_bar) + (19 * epsilon_theta_t) - (5 * epsilon_theta_t_plus_1) + (epsilon_theta_t_plus_2))

                   
                    sigma_t = eta * torch.sqrt((1 - alpha_t_prev) / (1 - alpha_t) * (1 - alpha_t / alpha_t_prev))
                    z = z_values[timestep].to(device)
                    image = (
                        torch.sqrt(alpha_t_prev / alpha_t) * image +
                        (torch.sqrt(1 - alpha_t_prev - sigma_t ** 2) - torch.sqrt(
                            (alpha_t_prev * (1 - alpha_t)) / alpha_t)) * z +
                        sigma_t * epsilon_t
                    )
                   
                    t_minus_1 = torch.full((image.shape[0],), time_steps[timestep-1], device=device, dtype=torch.long)
                    epsilon_theta_t_minus_1 = model(image, t_minus_1)

                    buffer.append((t_minus_1, epsilon_theta_t_minus_1))

                    # delta_epsilon = calculate_error_measure(epsilon_theta_t_minus_1, epsilon_t_minus_1_bar)
                   
                    print("error", delta_epsilon)
                j+=1
        
            
            if save_all_steps:
                images.append(image.cpu())
        print("count", count, new)

        return images if save_all_steps else image

