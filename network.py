import tensorflow as tf

def lrelu(X, leak=0.2):
    f1 = 0.5 * (1 + leak)
    f2 = 0.5 * (1 - leak)
    return f1 * X + f2 * tf.abs(X)

def generator(x, ngf=64, isTrain=True, reuse=False):
    with tf.variable_scope('generator', reuse=reuse):
        # initializer
        w_init = tf.truncated_normal_initializer(mean=0.0, stddev=0.02)
        b_init = tf.constant_initializer(0.0)

        # Unet encoder
        conv1 = tf.layers.conv2d(x, ngf, [4, 4], strides=(2, 2), padding='same', kernel_initializer=w_init, bias_initializer=b_init)
        conv2 = tf.layers.batch_normalization(tf.layers.conv2d(lrelu(conv1, 0.2), ngf * 2, [4, 4], strides=(2, 2), padding='same', kernel_initializer=w_init, bias_initializer=b_init), training=isTrain)
        conv3 = tf.layers.batch_normalization(tf.layers.conv2d(lrelu(conv2, 0.2), ngf * 4, [4, 4], strides=(2, 2), padding='same', kernel_initializer=w_init, bias_initializer=b_init), training=isTrain)
        conv4 = tf.layers.batch_normalization(tf.layers.conv2d(lrelu(conv3, 0.2), ngf * 8, [4, 4], strides=(2, 2), padding='same', kernel_initializer=w_init, bias_initializer=b_init), training=isTrain)
        conv5 = tf.layers.batch_normalization(tf.layers.conv2d(lrelu(conv4, 0.2), ngf * 8, [4, 4], strides=(2, 2), padding='same', kernel_initializer=w_init, bias_initializer=b_init), training=isTrain)
        conv6 = tf.layers.batch_normalization(tf.layers.conv2d(lrelu(conv5, 0.2), ngf * 8, [4, 4], strides=(2, 2), padding='same', kernel_initializer=w_init, bias_initializer=b_init), training=isTrain)
        conv7 = tf.layers.batch_normalization(tf.layers.conv2d(lrelu(conv6, 0.2), ngf * 8, [4, 4], strides=(2, 2), padding='same', kernel_initializer=w_init, bias_initializer=b_init), training=isTrain)
        conv8 = tf.layers.conv2d(lrelu(conv7, 0.2), ngf * 8, [4, 4], strides=(2, 2), padding='same', kernel_initializer=w_init, bias_initializer=b_init)

        # Unet decoder
        deconv1 = tf.nn.dropout(tf.layers.batch_normalization(tf.layers.conv2d_transpose(tf.nn.relu(conv8), ngf * 8, [4, 4], strides=(2, 2), padding='same', kernel_initializer=w_init, bias_initializer=b_init), training=isTrain), keep_prob=0.5)
        deconv1 = tf.concat([deconv1, conv7], 3)
        deconv2 = tf.nn.dropout(tf.layers.batch_normalization(tf.layers.conv2d_transpose(tf.nn.relu(deconv1), ngf * 8, [4, 4], strides=(2, 2), padding='same', kernel_initializer=w_init, bias_initializer=b_init), training=isTrain), keep_prob=0.5)
        deconv2 = tf.concat([deconv2, conv6], 3)
        deconv3 = tf.nn.dropout(tf.layers.batch_normalization(tf.layers.conv2d_transpose(tf.nn.relu(deconv2), ngf * 8, [4, 4], strides=(2, 2), padding='same', kernel_initializer=w_init, bias_initializer=b_init), training=isTrain), keep_prob=0.5)
        deconv3 = tf.concat([deconv3, conv5], 3)
        deconv4 = tf.layers.batch_normalization(tf.layers.conv2d_transpose(tf.nn.relu(deconv3), ngf * 8, [4, 4], strides=(2, 2), padding='same', kernel_initializer=w_init, bias_initializer=b_init), training=isTrain)
        deconv4 = tf.concat([deconv4, conv4], 3)
        deconv5 = tf.layers.batch_normalization(tf.layers.conv2d_transpose(tf.nn.relu(deconv4), ngf * 4, [4, 4], strides=(2, 2), padding='same', kernel_initializer=w_init, bias_initializer=b_init), training=isTrain)
        deconv5 = tf.concat([deconv5, conv3], 3)
        deconv6 = tf.layers.batch_normalization(tf.layers.conv2d_transpose(tf.nn.relu(deconv5), ngf * 2, [4, 4], strides=(2, 2), padding='same', kernel_initializer=w_init, bias_initializer=b_init), training=isTrain)
        deconv6 = tf.concat([deconv6, conv2], 3)
        deconv7 = tf.layers.batch_normalization(tf.layers.conv2d_transpose(tf.nn.relu(deconv6), ngf, [4, 4], strides=(2, 2), padding='same', kernel_initializer=w_init, bias_initializer=b_init), training=isTrain)
        deconv7 = tf.concat([deconv7, conv1], 3)
        deconv8 = tf.layers.conv2d_transpose(tf.nn.relu(deconv7), 3, [4, 4], strides=(2, 2), padding='same', kernel_initializer=w_init, bias_initializer=b_init)

        o = tf.nn.tanh(deconv8)

        return o

def discriminator(x, y, ndf=64, isTrain=True, reuse=False):
    with tf.variable_scope('discriminator', reuse=reuse):
        # initializer
        w_init = tf.truncated_normal_initializer(mean=0.0, stddev=0.02)
        b_init = tf.constant_initializer(0.0)

        cat1 = tf.concat([x, y], 3)
        conv1 = tf.layers.conv2d(cat1, ndf, [4, 4], strides=(2, 2), padding='same', kernel_initializer=w_init, bias_initializer=b_init)
        conv2 = tf.layers.batch_normalization(tf.layers.conv2d(lrelu(conv1, 0.2), ndf * 2, [4, 4], strides=(2, 2), padding='same', kernel_initializer=w_init, bias_initializer=b_init), training=isTrain)
        conv3 = tf.layers.batch_normalization(tf.layers.conv2d(lrelu(conv2, 0.2), ndf * 4, [4, 4], strides=(2, 2), padding='same', kernel_initializer=w_init, bias_initializer=b_init), training=isTrain)
        conv3 = tf.pad(conv3, [[0, 0], [1, 1], [1, 1], [0, 0]], mode="CONSTANT")
        conv4 = tf.layers.batch_normalization(tf.layers.conv2d(lrelu(conv3, 0.2), ndf * 8, [4, 4], strides=(1, 1), padding='valid', kernel_initializer=w_init, bias_initializer=b_init), training=isTrain)
        conv4 = tf.pad(conv4, [[0, 0], [1, 1], [1, 1], [0, 0]], mode="CONSTANT")
        conv5 = tf.layers.conv2d(lrelu(conv4, 0.2), 1, [4, 4], strides=(1, 1), padding='valid', kernel_initializer=w_init, bias_initializer=b_init)

        o = tf.nn.sigmoid(conv5)

        return o, conv5