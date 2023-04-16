from training import Model
from init import NUM_EPOCHS, SAVE_TIME, make_dirs, CHECKPOINT_PATH, CHI_LOAD_FILE, CONTINUE_TRAIN, BATCH_SIZE
from get_data import get_data
from evaluation import evaluate_model
from tqdm import tqdm
import torch
import numpy as np

make_dirs()

data_train, data_test, features_train, features_test = get_data()
model = Model()
losses_train = []
losses_test = []
chi_metric = []
epochs = range(NUM_EPOCHS)

if CONTINUE_TRAIN:
    checkpoint = torch.load(CHECKPOINT_PATH)
    last_epoch = checkpoint['epoch']
    model.generator.load_state_dict(checkpoint['gen_state_dict'])
    model.discriminator.load_state_dict(checkpoint['disc_state_dict'])
    model.disc_optimizer.load_state_dict(checkpoint['disc_optimizer_state_dict'])
    model.gen_optimizer.load_state_dict(checkpoint['gen_optimizer_state_dict'])

    with open(CHI_LOAD_FILE, 'rb') as f:
        chi_metric = np.load(f).tolist()

    epochs = range(last_epoch+1, NUM_EPOCHS)

for epoch in tqdm(epochs):
    cur_train_losses = model.train_epoch(data_train, features_train, epoch)
    losses_train.extend(cur_train_losses)

    cur_test_loss = model.test_epoch(data_test, features_test, epoch)
    losses_test.append(cur_test_loss)

    model.scheduler_step()

    if epoch % SAVE_TIME == 0:
        chi = evaluate_model(model, features_test, data_test, losses_train, losses_test, len(data_train), chi_metric, epoch)
        chi_metric.append(chi)
        torch.save({
            'epoch': epoch,
            'gen_state_dict': model.generator.state_dict(),
            'gen_optimizer_state_dict': model.gen_optimizer.state_dict(),
            'disc_state_dict': model.discriminator.state_dict(),
            'disc_optimizer_state_dict': model.disc_optimizer.state_dict(),
            'metric': chi,
            'train_losses': losses_train,
            'test_losses': losses_test
        }, 'models/checkpoint_' + str(epoch) + '.pth')
