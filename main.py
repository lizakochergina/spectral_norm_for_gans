from training import Model
from init import NUM_EPOCHS, device
from get_data import get_data
from evaluation import show_imgs, show_losses, show_metrics
from tqdm import tqdm
import torch

data_train, data_test, features_train, features_test = get_data()
gen_model = Model()
losses_train = []
losses_test = []

for epoch in tqdm(range(NUM_EPOCHS)):
    cur_train_losses = gen_model.train_epoch(data_train, features_train, epoch)
    losses_train.extend(cur_train_losses)

    cur_test_loss = gen_model.test_epoch(data_test, features_test, epoch)
    losses_test.append(cur_test_loss)

    gen_model.scheduler_step()

torch.save(gen_model.generator, 'generator.pth')
torch.save(gen_model.discriminator, 'discriminator.pth')

fake = gen_model.make_fake(torch.from_numpy(features_test).to(device)).cpu().detach().numpy()
show_imgs(data_test, fake)
show_losses(losses_train, losses_test, len(data_train))
show_metrics(data_test, fake)
