n_classes = 10
in_channels = 3
out_channels = 32

LATENT_DIM = 32
GP_LAMBDA = 10
BATCH_SIZE = 32
NUM_DISC_UPDATES = 8
NUM_EPOCHS = 101
SAVE_TIME = 5

pad_range = (-3, 5)
time_range = (-7, 9)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')