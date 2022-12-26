import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from init import pad_range, time_range


def show_losses(losses_train, losses_test, n):
    losses_train = pd.DataFrame(losses_train, columns=['epoch', 'sample', 'disc_loss', 'gen_loss'])
    losses_test = pd.DataFrame(losses_test, columns=['epoch', 'disc_loss', 'gen_loss'])
    plt.plot(losses_train.epoch + losses_train['sample'] / n, losses_train.disc_loss, label='disc, train')
    plt.plot(losses_test.epoch, losses_test.disc_loss, label='disc, test')
    plt.plot(losses_train.epoch + losses_train['sample'] / n, losses_train.gen_loss, label='gen, train')
    plt.plot(losses_test.epoch, losses_test.gen_loss, label='gen, test')
    plt.legend()
    plt.savefig('losses.png')


def gaussian_fit(img):
    assert img.ndim == 2
    assert (img >= 0).all()
    assert (img > 0).any()
    img_n = img / img.sum()

    mu = np.fromfunction(
        lambda i, j: (img_n[np.newaxis,...] * np.stack([i, j])).sum(axis=(1, 2)),
        shape=img.shape
    )
    cov = np.fromfunction(
        lambda i, j: (
            (img_n[np.newaxis,...] * np.stack([i * i, j * i, i * j, j * j])).sum(axis=(1, 2))
        ) - np.stack([mu[0]**2, mu[0]*mu[1], mu[0]*mu[1], mu[1]**2]),
        shape=img.shape
    ).reshape(2, 2)

    return mu, cov


def get_val_metric_single(img):
    assert img.ndim == 2
    img = np.where(img < 0, 0, img)
    mu, cov = gaussian_fit(img)
    # print(img)
    # print(mu)
    # print(cov)
    return np.array((*mu, *cov.diagonal(), cov[0, 1], img.sum()))


get_val_metric = np.vectorize(get_val_metric_single, signature='(m,n)->(k)')


def show_metrics(real_data, fake_data):
    metric_real = get_val_metric(real_data)
    metric_fake = get_val_metric(fake_data)
    plt.figure(figsize=(14, 8))
    labels = ['x0', 'x1', 'sigma0^2', 'sigma1^2', 'cov', 'amplitude']
    for i in range(6):
        plt.subplot(2, 3, i + 1)
        r = metric_real[:, i]
        f = metric_fake[:, i]
        bins = np.linspace(
            min(r.min(), f.min()),
            max(r.max(), f.max()),
            50
        )
        plt.hist(r, bins=bins, label='real', alpha=0.8)
        plt.hist(f, bins=bins, label='generated', alpha=0.5)
        plt.legend()
        plt.title(labels[i])
    plt.tight_layout()
    plt.savefig('metrics.png')


def show_imgs(real_data, fake_data):
    plt.figure(figsize=(17, 5))
    for i in range(5):
        plt.subplot(2, 5, i + 1)
        plt.imshow(fake_data[i, ::-1], extent=(*time_range, *pad_range))
        plt.title('generated')

    for i in range(5):
        plt.subplot(2, 5, i + 1 + 5)
        plt.imshow(real_data[i, ::-1], extent=(*time_range, *pad_range))
        plt.title('real')
    plt.savefig('imgs.png')