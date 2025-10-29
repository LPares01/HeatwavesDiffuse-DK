# %% [markdown]
# ## Simple example
# This notebook shows how to train model

# %%
# Imports
import sys
import matplotlib.pyplot as plt
import cartopy
import torch

sys.path.append('../src/')
from DatasetUS import *
from TrainDiffusion import *
from TrainUnet import *



# %% [markdown]
# This example can be run on a laptop but won't train the network very well. We will train with just a small subset of data. 

# %%
## Select years to train and validate
train_year_start = 1953
train_year_end = 1955

valid_year_start = 1956
valid_year_end = 1956

# %% [markdown]
# Set up training hyperparameters. We will only run for 10 epochs and we will use the cpu. 

# %%
## Select hyperparameters of training
batch_size = 8
learning_rate = 1e-4
accum = 8

# Run training for small number of epochs 
num_epochs = 100        

# Define device
device =  'cuda' if torch.cuda.is_available() else 'cpu'

# define the ml model
# unet_model = UNet((256, 128), 5, 3, label_dim=2, use_diffuse=False)
unet_model = UNet((64, 32), 5, 3, label_dim=2, use_diffuse=False)
unet_model.to(device)

# define the datasets
datadir = "../data/"
# dataset_train = UpscaleDataset(datadir, year_start=train_year_start, year_end=train_year_end,
#                                constant_variables=["lsm", "z"])

# dataset_test = UpscaleDataset(datadir, year_start=valid_year_start, year_end=valid_year_end,
#                               constant_variables=["lsm", "z"])

dataset_train = UpscaleDatasetDK(datadir, year_start=train_year_start, year_end=train_year_end,
                               constant_variables=["lsm", "z"])

dataset_test = UpscaleDatasetDK(datadir, year_start=valid_year_start, year_end=valid_year_end,
                              constant_variables=["lsm", "z"])

dataloader_train = torch.utils.data.DataLoader(
    dataset_train, batch_size=batch_size, shuffle=True, num_workers=4)
dataloader_test = torch.utils.data.DataLoader(
    dataset_test, batch_size=batch_size, shuffle=True, num_workers=4)

# %%
print(len(dataloader_train), len(dataloader_test))

# %%
scaler = torch.cuda.amp.GradScaler()

# define the optimiser
optimiser = torch.optim.AdamW(unet_model.parameters(), lr=learning_rate)

# Define the tensorboard writer

# writer = SummaryWriter("./runs_unet")
writer = SummaryWriter("./runs_unet_DK")

loss_fn = torch.nn.MSELoss()

# train the model
losses = []

# %% [markdown]
# Start the training loop. The plots generated will show the coarse res, the predicted, and the truth for a few samples and for different variables. At the start of training the first two columns (coarse res and predicted) look similar. Towards the end of the training, the last two columns (predicted and truth) should look similar. 

# %%
run_nbr = '02'
for step in range(num_epochs):
    epoch_loss = train_step(
        unet_model, loss_fn, dataloader_train, optimiser,
        scaler, step, accum, writer, device=device)
    losses.append(epoch_loss)

    if (step + 0) % 5 == 0:
        (fig, ax), (base_error, pred_error) = sample_model(
            unet_model, dataloader_test)
        fig.savefig(f"runs_unet_DK/run_{run_nbr}/epoch_{step}.png")
        plt.close(fig)

        writer.add_scalar("Error/base", base_error, step)
        writer.add_scalar("Error/pred", pred_error, step)

    # save the model
    if losses[-1] == min(losses):
        torch.save(unet_model.state_dict(), f"Models_unet_DK/run_{run_nbr}_epoch_{step}.pt")



