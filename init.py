import torch
import os

n_classes = 10
in_channels = 3
out_channels = 32

LATENT_DIM = 32
GP_LAMBDA = 10
BATCH_SIZE = 32
NUM_DISC_UPDATES = 8
NUM_EPOCHS = 2001
SAVE_TIME = 20

SPEC_NORM_DISC = True
SPEC_NORM_GEN = False

CHI_FILE = 'evaluation/chi_metric/chi_metric.npy'

LOSS = 'hinge'  # wasserstein or hinge

pad_range = (-3, 5)
time_range = (-7, 9)

metric_names = ['Mean0', 'Mean1', 'Sigma0^2', 'Sigma1^2', 'Cov01']

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def make_dirs():
    os.mkdir('models')
    os.mkdir('evaluation')
    os.mkdir('evaluation/losses')
    os.mkdir('evaluation/gaus_metrics')
    os.mkdir('evaluation/imgs')
    os.mkdir('evaluation/chi_plots')
    os.mkdir('evaluation/chi_metric')
