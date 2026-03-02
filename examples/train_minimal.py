# %% [markdown]
# ## Simple example
# This notebook shows how to train the model

# %%
# Imports
import os
import sys
import matplotlib.pyplot as plt
import torch
import wandb

sys.path.append(os.path.join(os.path.dirname(__file__), '../src/'))
from DatasetUS import *
from TrainDiffusion import *
from TrainUnet import *

# %%
# This example can be run on a laptop but won't train the network very well. We will train with just a small subset of data. 
# Train hyperparameters: we will only run for a few epochs and we will use the cpu. 

# Start a new wandb run to track this script.
run = wandb.init(
    # Set the wandb entity where your project will be logged (generally your team name).
    entity="lucp-masters-thesis",
    # Set the wandb project where this run will be logged.
    project="climate-diffuse",
    # Track hyperparameters and run metadata.
    config={
        "learning_rate": 1e-4,
        "architecture": "U-Net",
        "dataset": "minimal",
        "epochs": 100,
        "batch_size": 8,
        "accum": 16
    },
)
## Select years to train and validate
train_year_start = 1985
train_year_end = 1988

valid_year_start = 1989
valid_year_end = 1989

## Select hyperparameters of training
learning_rate = run.config["learning_rate"]
batch_size = run.config["batch_size"]

# Run training for small number of epochs 
num_epochs = run.config["epochs"]

# Define device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Running on', device)

# define the ml model
unet_model = UNet((256, 256), 3, 1, label_dim=1, use_diffuse=False)
unet_model.to(device)

# define the datasets
# datadir = os.path.join(os.path.dirname(__file__), "../data/CERRA-ERA5/dummy/")
datadir = os.path.join(os.path.dirname(__file__), "../data/CERRA-ERA5/")

dataset_train = UpscaleDatasetCERRA(datadir, year_start=train_year_start, year_end=train_year_end,
                               constant_variables=["lsm", "orog"])
dataset_val = UpscaleDatasetCERRA(datadir, year_start=valid_year_start, year_end=valid_year_end,
                              constant_variables=["lsm", "orog"])

dataloader_train = torch.utils.data.DataLoader(
    dataset_train, batch_size=batch_size, shuffle=True, num_workers=4)
dataloader_val = torch.utils.data.DataLoader(
    dataset_val, batch_size=batch_size, shuffle=True, num_workers=4)

# %%
print(len(dataloader_train), len(dataloader_val))

# %%
scaler = torch.amp.GradScaler("cuda") # type: ignore

# define the optimiser
optimiser = torch.optim.AdamW(unet_model.parameters(), lr=learning_rate)

run_nbr = '01'
run_path = os.path.join(os.path.dirname(__file__), f"./runs_unet_newCERRA/run_{run_nbr}")
os.makedirs(run_path, exist_ok=True)

# Define the tensorboard writer
# writer = SummaryWriter("./runs_unet")
# writer = SummaryWriter(run_path)

loss_fn = torch.nn.MSELoss()

# train the model
losses_val = []

# %% [markdown]
# Start the training loop. The plots generated will show the coarse res, the predicted, and the truth for a few samples and for different variables. At the start of training the first two columns (coarse res and predicted) look similar. Towards the end of the training, the last two columns (predicted and truth) should look similar. 

# %%

for step in range(num_epochs):

    epoch_loss = train_step(
        unet_model, loss_fn, dataloader_train, optimiser,
        scaler, step, run, device=device, logging_level='step')
    run.log({"Loss/train": epoch_loss})

    # evaluate on validation set every epoch
    val_loss = evaluate(unet_model, loss_fn, dataloader_val, device)
    losses_val.append(val_loss)
    run.log({"Loss/val": val_loss})

    # every few epochs: compute MAE and visualize feature, prediction and label on one validation batch 
    if (step + 1) % 1 == 0:
        (fig, ax), (base_error, pred_error) = sample_model(
            unet_model, dataloader_val, device)
        fig.savefig(run_path + f"/epoch_{step}.png")
        plt.close(fig)

        run.log({"MAE/base": base_error,
                 "MAE/val": pred_error})

    # save the model
    if losses_val[-1] == min(losses_val):
        torch.save(unet_model.state_dict(), f"{run_path}/run_{run_nbr}.pt")
        print(f"### New best model saved at epoch {step} with val_loss: {val_loss:.4f}")

run.finish()
print("Exited successfully!")
