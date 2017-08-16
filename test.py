import os, random
import numpy as np
import matplotlib.pylab as plt

file_list = os.listdir('data/cityscapes/train')
file_list.sort()
file_list = list(np.array(file_list)[random.sample(range(0, len(file_list)), len(file_list))])

img = plt.imread('data/cityscapes/train' + '/' + file_list[0])

a = 1