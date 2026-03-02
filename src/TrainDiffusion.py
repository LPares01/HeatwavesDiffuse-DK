import torch
from Network import EDMPrecond
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from DatasetUS import UpscaleDatasetCERRA
import time
import wandb
import os

# Loss class taken from EDS_Diffusion/loss.py
class EDMLoss:
    def __init__(self, P_mean=-1.2, P_std=1.2, sigma_data=1.0):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data

    def __call__(self, net, images, conditional_img=None, labels=None,
                 augment_pipe=None):
        rnd_normal = torch.randn([images.shape[0], 1, 1, 1], device=images.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data)**2
        y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
        n = torch.randn_like(y) * sigma
        D_yn = net(y + n, sigma, conditional_img, labels,
                   augment_labels=augment_labels)
        loss = weight * ((D_yn - y) ** 2)
        return loss


def train_step(model, loss_fn, data_loader, optimiser, scaler, step,
               run=None, device="cuda", logging_level='tqdm'):
    """
    Function for a single training step.
    :param model: instance of the Unet class
    :param loss_fn: loss function
    :param data_loader: data loader
    :param optimiser: optimiser to use
    :param scaler: scaler for mixed precision training
    :param step: current step
    :param run: wandb run entity
    :param device: device to use
    :param logging_level: amount of information to print out. 'tqdm' or 'step', otherwise no logging at all
    :return: loss value
    """

    start_time = time.time()

    model.train()

    disable_tqdm = (logging_level != 'tqdm')

    with tqdm(total=len(data_loader), dynamic_ncols=True, disable=disable_tqdm) as tq:
        tq.set_description(f"Train :: Epoch: {step}")

        epoch_losses = []
        step_loss = 0
        for i, batch in enumerate(data_loader):
            tq.update(1)

            image_input = batch["inputs"].to(device)
            image_output = batch["targets"].to(device)
            day = batch["doy"].to(device)
            # hour = batch["hour"].to(device)
            # condition_params = torch.stack((day, hour), dim=1)
            condition_params = day.unsqueeze(1)

            # forward diffusion
            with torch.amp.autocast("cuda"): # type: ignore
                loss = loss_fn(net=model, images=image_output,
                               conditional_img=image_input,
                               labels=condition_params)
                loss = torch.mean(loss)

            # backpropagation
            scaler.scale(loss).backward()
            step_loss += loss.item()

            accum = 4 if not run else run.config["accum"]
            if (i + 1) % accum == 0:
                scaler.step(optimiser)
                scaler.update()
                optimiser.zero_grad(set_to_none=True)

                step_loss = 0

            epoch_losses.append(loss.item())
            tq.set_postfix_str(s=f"Loss: {loss.item():.4f}")

            # if run is not None:
            #     run.log({"Loss/train": loss.item()})

        mean_loss = sum(epoch_losses) / len(epoch_losses)
        tq.set_postfix_str(s=f"Loss: {mean_loss:.4f}")

    if logging_level == 'step':
        step_time = time.time() - start_time
        print(f"Training step {step} finished after {step_time:.2f} seconds. Loss: {mean_loss:.4f}")

    return mean_loss


@torch.no_grad()
def sample_model(model, dataloader, num_steps=40, sigma_min=0.002,
                 sigma_max=80, rho=7, S_churn=40, S_min=0,
                 S_max=float('inf'), S_noise=1, device="cuda"):
    """
    Function for sampling the model.
    :param model: instance of the Unet class
    :param dataloader: data loader
    """

    model.eval() # added

    # Get n_images from the dataloader
    batch = next(iter(dataloader))
    images_input = batch["inputs"].to(device)
    coarse, fine = batch["coarse"], batch["fine"]

    # condition_params = torch.stack(
    #     (batch["doy"].to(device),
    #      batch["hour"].to(device)), dim=1)
    condition_params = batch["doy"].to(device).unsqueeze(1)
    
    sigma_min = max(sigma_min, model.sigma_min)
    sigma_max = min(sigma_max, model.sigma_max)

    init_noise = torch.randn((images_input.shape[0], 1, images_input.shape[2],
                              images_input.shape[3]),
                             dtype=torch.float64, device=device)

    # Time step discretization.
    step_indices = torch.arange(num_steps, dtype=torch.float64,
                                device=init_noise.device)
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1)
               * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat([model.round_sigma(t_steps),
                         torch.zeros_like(t_steps[:1])])  # t_N = 0

    # Main sampling loop.
    x_next = init_noise.to(torch.float64) * t_steps[0]
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])): # 0, ..., N-1
        x_cur = x_next

        # Increase noise temporarily.
        gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
        t_hat = model.round_sigma(t_cur + gamma * t_cur)
        x_hat = (x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise *
                 torch.randn_like(x_cur))

        # Euler step.
        denoised = model(x_hat, t_hat, images_input, condition_params).to(
            torch.float64)
        d_cur = (x_hat - denoised) / t_hat
        x_next = x_hat + (t_next - t_hat) * d_cur

        # Apply 2nd order correction.
        if i < num_steps - 1:
            denoised = model(x_next, t_next, images_input,
                             condition_params).to(torch.float64)
            d_prime = (x_next - denoised) / t_next
            x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

    predicted = dataloader.dataset.residual_to_fine_image(
        x_next.detach().cpu(), coarse)

    fig, _ = dataloader.dataset.plot_batch(coarse, fine, predicted)
    plt.subplots_adjust(wspace=0, hspace=0)

    return fig


@torch.no_grad()
def evaluate(model, loss_fn_val, dataloader, num_steps=40, sigma_min=0.002,
                 sigma_max=80, rho=7, S_churn=40, S_min=0,
                 S_max=float('inf'), S_noise=1, device="cuda"):
    """
    Function for sampling the model.
    :param model: instance of the Unet class
    :param dataloader: data loader
    """

    model.eval() # added

    val_losses = []
    for batch in dataloader:
        images_input = batch["inputs"].to(device)
        images_output = batch["targets"].to(device)
        day = batch["doy"].to(device)
        # hour = batch["hour"].to(device)
        # condition_params = torch.stack((day, hour), dim=1)
        condition_params = day.unsqueeze(1)
    
        sigma_min = max(sigma_min, model.sigma_min)
        sigma_max = min(sigma_max, model.sigma_max)

        init_noise = torch.randn((images_input.shape[0], 1, images_input.shape[2],
                                images_input.shape[3]),
                                dtype=torch.float64, device=device)

        # Time step discretization.
        step_indices = torch.arange(num_steps, dtype=torch.float64,
                                    device=init_noise.device)
        t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1)
                * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
        t_steps = torch.cat([model.round_sigma(t_steps),
                            torch.zeros_like(t_steps[:1])])  # t_N = 0

        # Main sampling loop.
        x_next = init_noise.to(torch.float64) * t_steps[0]
        for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])): # 0, ..., N-1
            x_cur = x_next

            # Increase noise temporarily.
            gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
            t_hat = model.round_sigma(t_cur + gamma * t_cur)
            x_hat = (x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise *
                    torch.randn_like(x_cur))

            # Euler step.
            denoised = model(x_hat, t_hat, images_input, condition_params).to(
                torch.float64)
            d_cur = (x_hat - denoised) / t_hat
            x_next = x_hat + (t_next - t_hat) * d_cur

            # Apply 2nd order correction.
            if i < num_steps - 1:
                denoised = model(x_next, t_next, images_input,
                                condition_params).to(torch.float64)
                d_prime = (x_next - denoised) / t_next
                x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)
        
        with torch.amp.autocast("cuda"): # type: ignore
            loss = loss_fn_val(x_next, images_output)
        
        val_losses.append(loss.item())
    
    return sum(val_losses) / len(val_losses)


def main():
    # Start a new wandb run to track this script.
    run = wandb.init(
        # Set the wandb entity where your project will be logged (generally your team name).
        entity="lucp-masters-thesis",
        # Set the wandb project where this run will be logged.
        project="climate-diffuse",
        # Track hyperparameters and run metadata.
        config={
            "architecture": "Diffusion",
            "dataset": "full",
            "epochs": 100,
            "learning_rate": 1e-4,
            "batch_size": 8,
            "accum": 16
        },
    )

    num_epochs = run.config["epochs"]
    learning_rate = run.config["learning_rate"]
    batch_size = run.config["batch_size"]

    # Define device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    network = EDMPrecond((256, 256), 4, 1, label_dim=1)
    network.to(device)

    # define the datasets
    dataset = run.config['dataset']

    if dataset == 'dummy':
        datadir = os.path.join(os.path.dirname(__file__), "../data/CERRA-ERA5/dummy/")
    else:
        datadir = os.path.join(os.path.dirname(__file__), "../data/CERRA-ERA5/")
    
    if dataset == 'full':
        train_start, train_end = 1985, 2008
        val_start, val_end = 2009, 2014
    else:
        train_start, train_end = 1985, 1988
        val_start, val_end = 1989, 1989

    dataset_train = UpscaleDatasetCERRA(datadir, year_start=train_start, year_end=train_end,
                                constant_variables=["lsm", "orog"])
    dataset_val = UpscaleDatasetCERRA(datadir, year_start=val_start, year_end=val_end,
                                constant_variables=["lsm", "orog"])

    dataloader_train = torch.utils.data.DataLoader(
        dataset_train, batch_size=batch_size, shuffle=True, num_workers=4)
    dataloader_val = torch.utils.data.DataLoader(
        dataset_val, batch_size=batch_size, shuffle=True, num_workers=4)

    scaler = torch.amp.GradScaler('cuda') 

    # define the optimiser
    optimiser = torch.optim.AdamW(network.parameters(), lr=learning_rate)

    # define the run directory
    run_nbr = '01'
    run_path = os.path.join(os.path.dirname(__file__), f"./runs_diffuse/run_{run_nbr}")
    os.makedirs(run_path, exist_ok=True)

    # define loss functions
    loss_fn = EDMLoss()
    loss_fn_val = torch.nn.MSELoss()

    # train the model
    losses_val = []
    for step in range(num_epochs):
        epoch_loss = train_step(
            network, loss_fn, dataloader_train, optimiser,
            scaler, step, run, device=device, logging_level='step')
        run.log({"Loss/train": epoch_loss})

        # evaluate on validation set every epoch
        val_loss = evaluate(network, loss_fn_val, dataloader_val, device=device)
        losses_val.append(val_loss)
        run.log({"Loss/val": val_loss})

        if (step + 1) % 1 == 0:
            fig = sample_model(network, dataloader_val, device=device)
            fig.savefig(run_path + f"/epoch_{step}.png")
            plt.close(fig)

        # save the model
        if losses_val[-1] == min(losses_val):
            torch.save(network.state_dict(), f"{run_path}/run_{run_nbr}.pt")
            print(f"### New best model saved at epoch {step} with val_loss: {val_loss:.4f}")

    run.finish()
    print("Exited successfully!")   

if __name__ == "__main__":
    main()
