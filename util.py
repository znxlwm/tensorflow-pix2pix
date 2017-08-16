import itertools, imageio, os, random
import matplotlib.pyplot as plt
import numpy as np
from scipy.misc import imresize

def show_result(x_, imgs, y_, num_epoch, show = False, save = False, path = 'result.png'):
    size_figure_grid = 3
    fig, ax = plt.subplots(x_.shape[0], size_figure_grid, figsize=(5, 5))
    for i, j in itertools.product(range(x_.shape[0]), range(size_figure_grid)):
        ax[i, j].get_xaxis().set_visible(False)
        ax[i, j].get_yaxis().set_visible(False)

    for i in range(x_.shape[0]):
        ax[i, 0].cla()
        ax[i, 0].imshow(denorm(x_[i]) / 255.0)
        ax[i, 1].cla()
        ax[i, 1].imshow(denorm(imgs[i]) / 255.0)
        ax[i, 2].cla()
        ax[i, 2].imshow(denorm(y_[i]) / 255.0)

    label = 'Epoch {0}'.format(num_epoch)
    fig.text(0.5, 0.04, label, ha='center')

    if save:
        plt.savefig(path)

    if show:
        plt.show()
    else:
        plt.close()

def show_train_hist(hist, show = False, save = False, path = 'Train_hist.png'):
    x = range(len(hist['D_losses']))

    y1 = hist['D_losses']
    y2 = hist['G_losses']

    plt.plot(x, y1, label='D_loss')
    plt.plot(x, y2, label='G_loss')

    plt.xlabel('Iter')
    plt.ylabel('Loss')

    plt.legend(loc=4)
    plt.grid(True)
    plt.tight_layout()

    if save:
        plt.savefig(path)

    if show:
        plt.show()
    else:
        plt.close()

def generate_animation(root, model, opt):
    images = []
    for e in range(opt.train_epoch):
        img_name = root + 'Fixed_results/' + model + str(e + 1) + '.png'
        images.append(imageio.imread(img_name))
    imageio.mimsave(root + model + 'generate_animation.gif', images, fps=5)

def imgs_resize(imgs, resize_scale = 286):
    outputs = np.zeros((imgs.shape[0], resize_scale, resize_scale, imgs.shape[3]))
    for i in range(imgs.shape[0]):
        img = imresize(imgs[i], [resize_scale, resize_scale])
        outputs[i] = img.astype(np.float32)

    return outputs

def random_crop(imgs1, imgs2, crop_size = 256):
    outputs1 = np.zeros((imgs1.shape[0], crop_size, crop_size, imgs1.shape[3]))
    outputs2 = np.zeros((imgs2.shape[0], crop_size, crop_size, imgs2.shape[3]))
    for i in range(imgs1.shape[0]):
        img1 = imgs1[i]
        img2 = imgs2[i]
        rand1 = np.random.randint(0, imgs1.shape[1] - crop_size)
        rand2 = np.random.randint(0, imgs2.shape[1] - crop_size)
        outputs1[i] = img1[rand1: crop_size + rand1, rand2: crop_size + rand2, :]
        outputs2[i] = img2[rand1: crop_size + rand1, rand2: crop_size + rand2, :]

    return outputs1, outputs2

def random_fliplr(imgs1, imgs2):
    outputs1 = np.zeros(imgs1.shape)
    outputs2 = np.zeros(imgs2.shape)
    for i in range(imgs1.shape[0]):
        if np.random.rand(1) < 0.5:
            outputs1[i] = np.fliplr(imgs1[i])
            outputs2[i] = np.fliplr(imgs2[i])
        else:
            outputs1[i] = imgs1[i]
            outputs2[i] = imgs2[i]

    return outputs1, outputs2

def norm(img):
    return (img - 127.5) / 127.5

def denorm(img):
    return (img * 127.5) + 127.5

class data_loader:
    def __init__(self, root, batch_size=1, shuffle=False):
        self.root = root
        self.batch_size = batch_size
        self.file_list = os.listdir(self.root)
        if shuffle:
            self.file_list = list(np.array(self.file_list)[random.sample(range(0, len(self.file_list)), len(self.file_list))])
        img = plt.imread(self.root + '/' + self.file_list[0])
        self.shape = (len(self.file_list), img.shape[0], img.shape[1], img.shape[2])
        self.flag = 0

    def next_batch(self):
        if self.flag + self.batch_size > self.shape[0]:
            self.file_list = list(np.array(self.file_list)[random.sample(range(0, len(self.file_list)), len(self.file_list))])
            self.flag = 0

        output = np.zeros((self.batch_size, self.shape[1], self.shape[2], self.shape[3]))
        temp = 0
        for i in range(self.flag, self.flag + self.batch_size):
            output[temp] = plt.imread(self.root + '/' + self.file_list[i])
            temp = temp + 1

        self.flag += self.batch_size

        return output