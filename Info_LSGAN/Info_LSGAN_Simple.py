import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

batch_size = 512
learning_rate = 1e-3
epoch = 10000
discrete_latent_size = 10
contin_latent_size = 2

x = tf.placeholder(tf.float32, shape=[None, 784])
x_image = tf.reshape(x, [-1, 28, 28, 1])
y_= tf.placeholder(tf.float32, shape=[None, 10])
z_in = tf.placeholder(tf.float32, shape=[batch_size, 112])
z_label_check = tf.slice(z_in, begin=[0, 100], size=[batch_size, 10])
z_contin_check = tf.slice(z_in, begin=[0, 110], size=[batch_size, 2])

initializer = tf.truncated_normal_initializer(stddev=0.02)


def int_to_onehot(z_label):

    one_hot_array = np.zeros(shape=[len(z_label), discrete_latent_size])
    one_hot_array[np.arange(len(z_label)), z_label] = 1

    return one_hot_array


def lrelu(x, leak=0.2, name="lrelu"):

    with tf.variable_scope(name):
        
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)

        return f1 * x + f2 * abs(x)


def generator(z):

    with tf.variable_scope("generator"):
       
        fc1 = tf.contrib.layers.fully_connected(inputs=z, num_outputs=7 * 7 * 128, activation_fn=tf.nn.relu,
                                                normalizer_fn=tf.contrib.layers.batch_norm,
                                                weights_initializer=initializer,
                                                scope="g_fc1")
        fc1 = tf.reshape(fc1, shape=[batch_size, 7, 7, 128])
        
        conv1 = tf.contrib.layers.conv2d(fc1, num_outputs=4 * 64, kernel_size=5, stride=1,
                                         padding="SAME", activation_fn=tf.nn.relu,
                                         normalizer_fn=tf.contrib.layers.batch_norm,
                                         weights_initializer=initializer,
                                         scope="g_conv1")
        conv1 = tf.reshape(conv1, shape=[batch_size, 14, 14, 64])

        conv2 = tf.contrib.layers.conv2d(conv1, num_outputs=4 * 32, kernel_size=5, stride=1,
                                         padding="SAME", activation_fn=tf.nn.relu,
                                         normalizer_fn=tf.contrib.layers.batch_norm,
                                         weights_initializer=initializer,
                                         scope="g_conv2")
        conv2 = tf.reshape(conv2, shape=[batch_size, 28, 28, 32])
        
        conv3 = tf.contrib.layers.conv2d(conv2, num_outputs=1, kernel_size=5, stride=1,
                                         padding="SAME", activation_fn=tf.nn.tanh,
                                         scope="g_conv3")

        return conv3


def discriminator(tensor, reuse=False):
    
    with tf.variable_scope("discriminator"):

        conv1 = tf.contrib.layers.conv2d(inputs=tensor, num_outputs=32, kernel_size=5, stride=2,
                                         padding="SAME",reuse=reuse, activation_fn=lrelu,
                                         weights_initializer=initializer,
                                         scope="d_conv1")

        conv2 = tf.contrib.layers.conv2d(inputs=conv1, num_outputs=64, kernel_size=5, stride=2,
                                         padding="SAME", reuse=reuse, activation_fn=lrelu,
                                         normalizer_fn=tf.contrib.layers.batch_norm,
                                         weights_initializer=initializer,
                                         scope="d_conv2")

        fc1 = tf.reshape(conv2, shape=[batch_size, 7 * 7 * 64])
        fc1 = tf.contrib.layers.fully_connected(inputs=fc1, num_outputs=512,reuse=reuse, activation_fn=lrelu,
                                                normalizer_fn=tf.contrib.layers.batch_norm,
                                                weights_initializer=initializer,
                                                scope="d_fc1")

        label_output = tf.contrib.layers.fully_connected(inputs=fc1, num_outputs=discrete_latent_size,
                                                         reuse=reuse, activation_fn=lrelu,
                                                         weights_initializer=initializer,
                                                         scope="d_label")
        contin_output = tf.contrib.layers.fully_connected(inputs=fc1, num_outputs=contin_latent_size,
                                                          reuse=reuse, activation_fn=tf.nn.tanh,
                                                          weights_initializer=initializer,
                                                          scope="d_contin")
        output = tf.contrib.layers.fully_connected(inputs=fc1, num_outputs=1, reuse=reuse,
                                                   activation_fn=tf.nn.sigmoid,
                                                   weights_initializer=initializer,
                                                   scope="d_output")

        return output, label_output, contin_output


g_out = generator(z_in)
d_out_fake, label_fake, contin_fake = discriminator(g_out)
d_out_real, label_real, contin_real = discriminator(x_image,reuse=True)

# Loss Calculation

disc_loss = tf.reduce_sum(tf.square(d_out_real-1) + tf.square(d_out_fake))/2
gen_loss = tf.reduce_sum(tf.square(d_out_fake-1))/2

disc_label_loss = tf.reduce_sum(tf.losses.softmax_cross_entropy(label_real, y_))
gen_label_loss = tf.reduce_sum(tf.losses.softmax_cross_entropy(label_fake, z_label_check))

#disc_contin_loss = tf.losses.mean_squared_error(tf.losses.softmax_cross_entropy())
gen_contin_loss = tf.losses.mean_squared_error(z_contin_check, contin_fake)

disc_loss_total = disc_loss + disc_label_loss
gen_loss_total = gen_loss + gen_label_loss + gen_contin_loss

# Optimizer & Apply Gradients

gen_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="generator") 
dis_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="discriminator") 

d_optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate)
g_optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate)

d_grads = d_optimizer.compute_gradients(disc_loss_total, dis_variables)
g_grads = g_optimizer.compute_gradients(gen_loss_total, gen_variables)

update_D = d_optimizer.apply_gradients(d_grads)
update_G = g_optimizer.apply_gradients(g_grads)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    for i in range(epoch):
        batch = mnist.train.next_batch(batch_size)

        z_random = np.random.uniform(0, 1.0, size=[batch_size, 100]).astype(np.float32)
        z_label = np.random.randint(0, 10, size=batch_size)
        z_label_onehot = int_to_onehot(z_label)
        z_contin = 2 * np.random.random(size=[batch_size,2]) - 1
        z_concat = np.concatenate([z_random, z_label_onehot,z_contin], axis=1)

        _, d_loss = sess.run([update_D, disc_loss_total], feed_dict={x: batch[0], y_:batch[1], z_in: z_concat})
        
        for j in range(5):
            _, g_loss = sess.run([update_G, gen_loss_total], feed_dict={z_in: z_concat})

        print("i: {} / d_loss: {} / g_loss: {}".format(i, np.sum(d_loss)/batch_size, np.sum(g_loss)/batch_size))

        if i % 100 == 0:

            gen_o = sess.run(g_out, feed_dict={z_in: z_concat})

            for k in range(10):
                plt.imsave("{}th_[{},{}]_{}.png".format(i, round(z_contin[k][0], 2), round(z_contin[k][1], 2), z_label[k]), gen_o[k][:, :, 0], cmap="gray")
