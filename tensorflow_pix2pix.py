import os, time, pickle, argparse, network, util
import tensorflow as tf
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=False, default='facades',  help='')
parser.add_argument('--train_subfolder', required=False, default='train',  help='')
parser.add_argument('--test_subfolder', required=False, default='val',  help='')
parser.add_argument('--batch_size', type=int, default=1, help='train batch size')
parser.add_argument('--test_batch_size', type=int, default=5, help='test batch size')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--input_size', type=int, default=256, help='input size')
parser.add_argument('--crop_size', type=int, default=256, help='crop size (0 is false)')
parser.add_argument('--resize_scale', type=int, default=286, help='resize scale (0 is false)')
parser.add_argument('--fliplr', type=bool, default=True, help='random fliplr True or False')
parser.add_argument('--train_epoch', type=int, default=200, help='number of train epochs')
parser.add_argument('--lrD', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--lrG', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--L1_lambda', type=float, default=100, help='lambda for L1 loss')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for Adam optimizer')
parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for Adam optimizer')
parser.add_argument('--save_root', required=False, default='results', help='results save path')
parser.add_argument('--inverse_order', type=bool, default=True, help='0: [input, target], 1 - [target, input]')
opt = parser.parse_args()
print(opt)

# results save path
root = opt.dataset + '_' + opt.save_root + '/'
model = opt.dataset + '_'
if not os.path.isdir(root):
    os.mkdir(root)
if not os.path.isdir(root + 'Fixed_results'):
    os.mkdir(root + 'Fixed_results')

# data_loader
train_loader = util.data_loader('data/' + opt.dataset + '/' + opt.train_subfolder, opt.batch_size, shuffle=True)
test_loader = util.data_loader('data/' + opt.dataset + '/' + opt.test_subfolder, opt.test_batch_size, shuffle=True)
img_size = train_loader.shape[1]
test_img = test_loader.next_batch()
if opt.inverse_order:
    fixed_x_ = test_img[:, :, img_size:, :]
    fixed_y_ = test_img[:, :, 0:img_size, :]
else:
    fixed_x_ = test_img[:, :, 0:img_size, :]
    fixed_y_ = test_img[:, :, img_size:, :]

if img_size != opt.input_size:
    fixed_x_ = util.imgs_resize(fixed_x_, opt.input_size)
    fixed_y_ = util.imgs_resize(fixed_y_, opt.input_size)

fixed_x_ = util.norm(fixed_x_)
fixed_y_ = util.norm(fixed_y_)
# variables
x = tf.placeholder(tf.float32, shape=(None, opt.input_size, opt.input_size, train_loader.shape[3]))
y = tf.placeholder(tf.float32, shape=(None, opt.input_size, opt.input_size, train_loader.shape[3]))

# network
G = network.generator(x, opt.ngf)
D_positive, D_positive_logits = network.discriminator(x, y, opt.ndf)
D_negative, D_negative_logits = network.discriminator(x, G, opt.ndf, reuse=True)

# loss
D_loss_positive = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_positive_logits, labels=tf.ones_like(D_positive_logits)))
D_loss_negative = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_negative_logits, labels=tf.zeros_like(D_negative_logits)))
D_loss = (D_loss_positive + D_loss_negative) * 0.5
G_loss_gan = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_negative_logits, labels=tf.ones_like(D_negative_logits)))
G_loss_L1 = tf.reduce_mean(tf.reduce_sum(tf.abs(G - y), 3))
G_loss = G_loss_gan + opt.L1_lambda * G_loss_L1

# trainable variables for each network
T_vars = tf.trainable_variables()
D_vars = [var for var in T_vars if var.name.startswith('discriminator')]
G_vars = [var for var in T_vars if var.name.startswith('generator')]

# Adam optimizer
with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
    D_optim = tf.train.AdamOptimizer(opt.lrD, beta1=opt.beta1, beta2=opt.beta2).minimize(D_loss, var_list=D_vars)
    G_optim = tf.train.AdamOptimizer(opt.lrG, beta1=opt.beta1, beta2=opt.beta2).minimize(G_loss, var_list=G_vars)

# open session and initialize all variables
saver = tf.train.Saver()
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

train_hist = {}
train_hist['D_losses'] = []
train_hist['G_losses'] = []
train_hist['per_epoch_ptimes'] = []
train_hist['total_ptime'] = []

print('training start!')
start_time = time.time()
for epoch in range(opt.train_epoch):
    D_losses = []
    G_losses = []
    epoch_start_time = time.time()
    num_iter = 0
    for iter in range(train_loader.shape[0] // opt.batch_size):
        train_img = train_loader.next_batch()

        if opt.inverse_order:
            x_ = train_img[:, :, img_size:, :]
            y_ = train_img[:, :, 0:img_size, :]
        else:
            x_ = train_img[:, :, 0:img_size, :]
            y_ = train_img[:, :, img_size:, :]

        if img_size != opt.input_size:
            x_ = util.imgs_resize(x_, opt.input_size)
            y_ = util.imgs_resize(y_, opt.input_size)

        if opt.resize_scale:
            x_ = util.imgs_resize(x_, opt.resize_scale)
            y_ = util.imgs_resize(y_, opt.resize_scale)

        if opt.crop_size:
            x_, y_ = util.random_crop(x_, y_, opt.crop_size)

        if opt.fliplr:
            x_, y_ = util.random_fliplr(x_, y_)

        x_ = util.norm(x_)
        y_ = util.norm(y_)

        loss_d_, _ = sess.run([D_loss, D_optim], {x: x_, y: y_})
        D_losses.append(loss_d_)
        train_hist['D_losses'].append(loss_d_)

        loss_g_, _ = sess.run([G_loss, G_optim], {x: x_, y: y_})
        G_losses.append(loss_g_)
        train_hist['G_losses'].append(loss_g_)

        num_iter += 1

    epoch_end_time = time.time()
    per_epoch_ptime = epoch_end_time - epoch_start_time

    print('[%d/%d] - ptime: %.2f, loss_d: %.3f, loss_g: %.3f' % ((epoch + 1), opt.train_epoch, per_epoch_ptime, np.mean((D_losses)), np.mean(G_losses)))
    fixed_p = root + 'Fixed_results/' + model + str(epoch + 1) + '.png'
    outputs = sess.run(G, {x: fixed_x_})
    util.show_result(fixed_x_, outputs, fixed_y_, (epoch+1), save=True, path=fixed_p)
    train_hist['per_epoch_ptimes'].append(per_epoch_ptime)

end_time = time.time()
total_ptime = end_time - start_time
train_hist['total_ptime'].append(total_ptime)

print("Avg. one epoch ptime: %.2f, total %d epochs ptime: %.2f" % (np.mean(train_hist['per_epoch_ptimes']), opt.train_epoch, total_ptime))
print("Training finish!... save training results")
with open(root + model + 'train_hist.pkl', 'wb') as f:
    pickle.dump(train_hist, f)

saver.save(sess, root + model + 'params.ckpt')

util.show_train_hist(train_hist, save=True, path=root + model + 'train_hist.png')
util.generate_animation(root, model, opt)
