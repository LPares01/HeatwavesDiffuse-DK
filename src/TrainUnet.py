import torch
from torch.nn.modules.module import Module
from DatasetUS import UpscaleDatasetCERRA
import matplotlib.pyplot as plt
from Network import UNet
from tqdm import tqdm
import time
import wandb
import os

class WeightedMSELoss(Module):
    """
    MSE Loss with higher weights for extreme values (above a given percentile). The higher the percentile, the higher the weights.
    Adapted from Jia, 2024.
    """
    def __init__(self, dataset, percentile=95):
        super().__init__()
        self.percentile = percentile
        self.weight_factor = percentile/(100-percentile)
        # Compute thresholds for extreme values
        self.register_buffer('thresholds', torch.quantile(dataset.fine, self.percentile/100, dim=0))
        print(f"WeightedMSELoss initialized with thresholds at {percentile}%")

    def forward(self, pred, target):
        thresholds = self.thresholds

        # Create weights: higher for values above threshold
        weights = torch.where(
            target >= thresholds,
            torch.tensor(self.weight_factor, device=target.device),
            torch.tensor(1, device=target.device)
        )
        
        # Weighted MSE
        loss = weights * (pred - target) ** 2
        return loss.mean()
    
class MaskedMSELoss(Module):
    """
    MSE Loss ignoring masked values.
    """
    def __init__(self, dataset):
        super().__init__()
        self.register_buffer('mask', dataset.mask == False)
        self.nb_mask = self.mask.sum()
        print(f"MaskedMSELoss initialized with {self.nb_mask} valid pixels.")

    def forward(self, pred, target):        
        # Masked MSE
        loss = (pred - target) ** 2
        return (loss * self.mask).sum() / self.nb_mask
    
class MaskedWMSELoss(Module):
    """
    WMSE with sea-masking.
    """
    def __init__(self, dataset, percentile=95):
        super().__init__()

        # Thresholds for extreme values
        self.percentile = percentile
        self.weight_factor = percentile/(100-percentile)
        self.register_buffer('thresholds', torch.quantile(dataset.fine, self.percentile/100, dim=0))

        # Sea-masking
        self.register_buffer('mask', dataset.mask == False)
        self.nb_mask = self.mask.sum()

        print(f"WeightedMSELoss initialized with thresholds at {percentile}% and {self.nb_mask} valid pixels.")

    def forward(self, pred, target):
        thresholds = self.thresholds

        # Create weights: higher for values above threshold, 0 for sea pixels, 1 for other values
        weights = torch.where(
            target >= thresholds,
            torch.tensor(self.weight_factor, device=target.device),
            torch.tensor(1, device=target.device)
        ) * self.mask

        print(f"Extreme pixels: {(weights > 1).numel()} / {weights.numel()}")
        print(f"Sea pixels: {(weights == 0).numel()} / {weights.numel()}")
        print(f"Mean weight: {weights.mean().item():.3f}")
        
        # Weighted MSE
        loss = weights * (pred - target) ** 2
        return loss.sum() / self.nb_mask
    

def train_step(model, loss_fn, data_loader, optimiser, scaler, step,
               run, device="cuda", logging_level='tqdm'):
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

        accum = run.config["accum"]

        bp_times = []
        accum_times = []
        epoch_losses = []
        for i, batch in enumerate(data_loader):
            tq.update(1)
            image_input = batch["inputs"].to(device)
            image_output = batch["targets"].to(device)
            condition_params = torch.cat(
                (batch["doy"].to(device),
                batch["year"].unsqueeze(1).to(device)), dim=1)

            # forward unet
            with torch.amp.autocast("cuda"): # type: ignore
                model_out = model(image_input,
                                  class_labels=condition_params)
                loss = loss_fn(model_out, image_output)

            # backpropagation
            bp_start = time.time()
            scaler.scale(loss / accum).backward()
            bp_times.append(time.time() - bp_start)

            if (i + 1) % accum == 0:
                accum_start = time.time()
                scaler.step(optimiser)
                scaler.update()
                optimiser.zero_grad(set_to_none=True)
                accum_times.append(time.time() - accum_start)

            epoch_losses.append(loss.item())
            tq.set_postfix_str(s=f"Loss: {loss.item():.4f}")

        mean_loss = sum(epoch_losses) / len(epoch_losses)
        tq.set_postfix_str(s=f"Loss: {mean_loss:.4f}")

    if logging_level == 'step':
        step_time = time.time() - start_time
        print(f"Training step {step} finished after {step_time:.2f} seconds (incl. {sum(bp_times):.2f} spent\nbackpropagating and {sum(accum_times):.2f} between accumulations). Loss: {mean_loss:.4f}")

    return mean_loss


@torch.no_grad()
def sample_model(model, dataloader, device="cuda"):
    """
    Function for sampling the model.
    :param model: instance of the Unet class
    :param dataloader: data loader
    """

    model.eval()

    # Get n_images from the dataloader
    batch = next(iter(dataloader))
    images_input = batch["inputs"].to(device)
    coarse, fine = batch["coarse"], batch["fine"]
    condition_params = torch.cat(
                (batch["doy"].to(device),
                batch["year"].unsqueeze(1).to(device)), dim=1)
    residual = model(images_input, class_labels=condition_params)

    predicted = dataloader.dataset.residual_to_fine_image(
        residual.detach().cpu(), coarse)

    fig, _ = dataloader.dataset.plot_batch(coarse, fine, predicted)
    plt.subplots_adjust(wspace=0, hspace=0)

    return fig

# Added to enable saving the model based on minimal validation loss instead of training loss -> avoids overfitting
@torch.no_grad()
def evaluate(model, loss_fn, dataloader, device="cuda"):
    """
    Evaluate model on validation set.
    :param model: instance of the Unet class
    :param loss_fn: loss function
    :param data_loader: validation data loader
    :param device: device to use
    :return: mean validation loss
    """

    model.eval()
    
    val_losses = []
    for batch in dataloader:
        image_input = batch["inputs"].to(device)
        image_output = batch["targets"].to(device)
        condition_params = torch.cat(
                (batch["doy"].to(device),
                batch["year"].unsqueeze(1).to(device)), dim=1)

        with torch.amp.autocast("cuda"): # type: ignore
            model_out = model(image_input, class_labels=condition_params)
            loss = loss_fn(model_out, image_output)
        
        val_losses.append(loss.item())
    
    return sum(val_losses) / len(val_losses)


def main():
    # torch.backends.cudnn.benchmark = True
    # Start a new wandb run to track this script.
    run = wandb.init(
        # Set the wandb entity where your project will be logged (generally your team name).
        entity="lucp-masters-thesis",
        # Set the wandb project where this run will be logged.
        project="climate-diffuse",
        # Track hyperparameters and run metadata.
        config={
            "architecture": "unet",
            "dataset": "dummy", # 'full', 'minimal' or 'dummy'
            "epochs": 1,
            "learning_rate": 1e-4,
            "batch_size": 2,
            "accum": 2,
            "loss_fn": "Weighted", # 'Weighted' or 'MSE'
            "percentile": 95,
            "sea_mask": True,
            "save_checkpoint": False,
            "load_checkpoint": False,
            "run_ID": '07'
        },
    )
    run_nbr = run.config['run_ID']

    num_epochs = run.config["epochs"]
    learning_rate = run.config["learning_rate"]
    batch_size = run.config["batch_size"]

    # define device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # define the ml model
    unet_model = UNet((256, 256), 3, 1, label_dim=3, use_diffuse=False)
    unet_model.to(device)

    # define the datasets
    dataset = run.config['dataset']

    if dataset == 'dummy':
        datadir = os.path.join(os.path.dirname(__file__), "../data/CERRA-ERA5/dummy/")
        logging_level = 'tqdm'
    else:
        logging_level = 'step'
        datadir = os.path.join(os.path.dirname(__file__), "../data/CERRA-ERA5/")
    
    if dataset == 'full':
        train_start, train_end = 1985, 2008
        val_start, val_end = 2009, 2014
    else:
        train_start, train_end = 1985, 1988
        val_start, val_end = 1989, 1989

    dataset_train = UpscaleDatasetCERRA(datadir, year_start=train_start, year_end=train_end,
                                constant_variables=["lsm", "orog"], sea_masking=run.config['sea_mask'])
    dataset_val = UpscaleDatasetCERRA(datadir, year_start=val_start, year_end=val_end,
                                constant_variables=["lsm", "orog"], sea_masking=run.config['sea_mask'])

    # dataset_train.to(device)
    # dataset_val.to(device)
    # print("Datasets loaded to GPU")

    dataloader_train = torch.utils.data.DataLoader(
        dataset_train, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True)
    dataloader_val = torch.utils.data.DataLoader(
        dataset_val, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True)

    scaler = torch.amp.GradScaler('cuda')  # type: ignore

    # define the optimiser
    optimiser = torch.optim.AdamW(unet_model.parameters(), lr=learning_rate)

    # define the run directory
    run_path = os.path.join(os.path.dirname(__file__), f"./runs_unet/run_{run_nbr}")
    os.makedirs(run_path, exist_ok=True)

    if os.path.exists(f"{run_path}/run_{run_nbr}_chk.pt") and run.config["load_checkpoint"]:
        print("Loading model from checkpoint...")
        # load the model checkpoint
        checkpoint = torch.load(f"{run_path}/run_{run_nbr}_chk.pt", weights_only=False)
        # load model weights state_dict
        unet_model.load_state_dict(checkpoint['model'])
        optimiser.load_state_dict(checkpoint['optimizer'])
        scaler.load_state_dict(checkpoint['scaler'])
        name_loss_fn = checkpoint['loss_fn']
        sea_masking = checkpoint['sea_mask']
        print('Previously trained model weights, optimizer and scaler state_dicts loaded.')
        done_epochs = checkpoint['epoch']+1
        losses_val = [checkpoint['best_loss']]
        print(f"Previously trained for {done_epochs} epochs, with best validation loss: {losses_val[0]:.4f}")
        # train for more epochs
        epoch_start = done_epochs
        print(f"Train for {num_epochs - epoch_start} more epochs.")
    else:
        if run.config["load_checkpoint"]:
            print(f"load_checkpoint is True but no checkpoint found at {run_path}/run_{run_nbr}_chk.pt. Running from scratch.")
        epoch_start = 0
        name_loss_fn = run.config["loss_fn"]
        sea_masking = run.config['sea_mask']
        losses_val = []

    # define the loss function
    if name_loss_fn == 'Weighted':
        p = run.config["percentile"]
        if sea_masking:
            loss_fn = MaskedWMSELoss(dataset_train, percentile=p)
        else:
            loss_fn = WeightedMSELoss(dataset_train, percentile=p)
    else:
        if sea_masking:
            loss_fn = MaskedMSELoss(dataset_train)
        else:
            loss_fn = torch.nn.MSELoss()
    loss_fn.to(device)

    # train the model
    val_interval = 5
    for step in range(epoch_start, num_epochs):
        epoch_loss = train_step(
            unet_model, loss_fn, dataloader_train, optimiser,
            scaler, step, run, device=device, logging_level=logging_level)
        run.log({"Loss/train": epoch_loss})

        # evaluate on validation set every 5 epochs except when the previous validation loss was better (in which case, evaluate every epoch)
        if step % val_interval == 0 and step != num_epochs - 1:
            val_loss = evaluate(unet_model, loss_fn, dataloader_val, device)
            losses_val.append(val_loss)
            run.log({"Loss/val": val_loss})
            # save the model if it performs better
            if losses_val[-1] == min(losses_val):
                # save model weights only
                torch.save(unet_model.state_dict(), f"{run_path}/run_{run_nbr}.pt")
                print(f"### New best model saved at epoch {step} with val_loss: {val_loss:.4f}")
                val_interval = 1
            else:
                val_interval = 5

        # sample model every 10 epochs
        if step % 10 == 0:
            fig = sample_model(unet_model, dataloader_val, device)
            fig.savefig(run_path + f"/epoch_{step}.png")
            plt.close(fig)

        # last epoch: evaluate the model and save checkpoint or weights
        if step == num_epochs - 1:
            val_loss = evaluate(unet_model, loss_fn, dataloader_val, device)
            losses_val.append(val_loss)
            run.log({"Loss/val": val_loss})
            # save the model if it performs better
            if losses_val[-1] == min(losses_val):
                # save model weights only
                torch.save(unet_model.state_dict(), f"{run_path}/run_{run_nbr}.pt")
                print(f"### New best model saved at epoch {step} with val_loss: {val_loss:.4f}")
            if run.config["save_checkpoint"]:
                # save model checkpoint
                torch.save({
                    'epoch': step,
                    'model': unet_model.state_dict(),
                    'optimizer': optimiser.state_dict(),
                    'scaler': scaler.state_dict(),
                    'loss_fn': name_loss_fn,
                    'best_loss': min(losses_val),
                    'sea_mask': sea_masking
                }, f"{run_path}/run_{run_nbr}_chk.pt")
                print(f'Model checkpoint saved at: {run_path}/run_{run_nbr}_chk.pt')
       
    run.finish()
    print("Exited successfully!")   

if __name__ == "__main__":
    main()