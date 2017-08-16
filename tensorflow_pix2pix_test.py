import os, time, argparse, network, util
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=False, default='facades',  help='')
parser.add_argument('--input_size', type=int, default=256, help='input image size')
parser.add_argument('--test_subfolder', required=False, default='val',  help='')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--save_root', required=False, default='results', help='results save path')
parser.add_argument('--inverse_order', type=bool, default=True, help='True: [input, target], False: [target, input]')
opt = parser.parse_args()
print(opt)

# results save path
if not os.path.isdir(opt.dataset + '_results/test_results'):
    os.mkdir(opt.dataset + '_results/test_results')

# data_loader
test_loader = util.data_loader('data/' + opt.dataset + '/' + opt.test_subfolder, 1, shuffle=False)
img_size = test_loader.shape[1]

# variables
x = tf.placeholder(tf.float32, shape=(None, opt.input_size, opt.input_size, test_loader.shape[3]))

# network
G = network.generator(x, opt.ngf)

# open session and initialize all variables
saver = tf.train.Saver()
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
saver.restore(sess, tf.train.latest_checkpoint(opt.dataset + '_results'))

print('test start!')

per_ptime = []
total_start_time = time.time()
for iter in range(test_loader.shape[0]):
    per_start_time = time.time()

    train_img = test_loader.next_batch()

    if opt.inverse_order:
        x_ = train_img[:, :, img_size:, :]
        y_ = train_img[:, :, 0:img_size, :]
    else:
        x_ = train_img[:, :, 0:img_size, :]
        y_ = train_img[:, :, img_size:, :]

    if img_size != opt.input_size:
        x_ = util.imgs_resize(x_, opt.input_size)
        y_ = util.imgs_resize(y_, opt.input_size)

    x_ = util.norm(x_)
    y_ = util.norm(y_)

    test_img = sess.run(G, {x: x_})

    num_str = test_loader.file_list[iter][:test_loader.file_list[iter].find('.')]
    path = opt.dataset + '_results/test_results/' + num_str + '_input.png'
    plt.imsave(path, (util.denorm(x_[0]) / 255))
    path = opt.dataset + '_results/test_results/' + num_str + '_output.png'
    plt.imsave(path, (util.denorm(test_img[0]) / 255))
    path = opt.dataset + '_results/test_results/' + num_str + '_target.png'
    plt.imsave(path, (util.denorm(y_[0]) / 255))

    per_end_time = time.time()
    per_ptime.append(per_end_time - per_start_time)

total_end_time = time.time()
total_ptime = total_end_time - total_start_time

print('total %d images generation complete!' % (iter+1))
print('Avg. one image process ptime: %.2f, total %d images process ptime: %.2f' % (np.mean(per_ptime), (iter+1), total_ptime))