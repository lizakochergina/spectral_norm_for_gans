from training import Model
from init import NUM_EPOCHS, SAVE_TIME, device, make_dirs
from get_data import get_data
from evaluation import evaluate_model, show_imgs, show_losses, show_metrics
from tqdm import tqdm
import torch

make_dirs()

data_train, data_test, features_train, features_test = get_data()
gen_model = Model()
losses_train = []
losses_test = []
chi_metric = []

for epoch in tqdm(range(NUM_EPOCHS)):
    cur_train_losses = gen_model.train_epoch(data_train, features_train, epoch)
    losses_train.extend(cur_train_losses)

    cur_test_loss = gen_model.test_epoch(data_test, features_test, epoch)
    losses_test.append(cur_test_loss)

    gen_model.scheduler_step()

    if epoch % SAVE_TIME == 0:
        chi = evaluate_model(gen_model, features_test, data_test, losses_train, losses_test, len(data_train), chi_metric, epoch)
        chi_metric.append(chi)
        torch.save({
            'epoch': epoch,
            'model_state_dict': gen_model.generator.state_dict(),
            'metric': chi,
        }, 'models/generator_' + str(epoch) + '.pth')
