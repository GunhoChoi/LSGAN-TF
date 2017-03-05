import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

batch_size = 512
learning_rate = 1e-3
epoch = 10000

x = tf.placeholder(tf.float32, shape=[None, 784])
x_image = tf.reshape(x, [-1, 28, 28, 1])
z_in = tf.placeholder(tf.float32, shape=[batch_size, 100])

initializer = tf.truncated_normal_initializer(stddev=0.02)


def lrelu(x, leak=0.2, name="lrelu"):
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)

        return f1 * x + f2 * abs(x)


def generator(z):
    
    with tf.variable_scope("generator"):
        fc1 = tf.contrib.layers.fully_connected(inputs=z, num_outputs=7*7*128, activation_fn=tf.nn.relu, \
                                                normalizer_fn=tf.contrib.layers.batch_norm,\
                                                weights_initializer=initializer,scope="g_fc1")

        fc1 = tf.reshape(fc1, shape=[batch_size, 7, 7, 128])
        
        conv1 = tf.contrib.layers.conv2d(fc1, num_outputs=4*64, kernel_size=5, stride=1, padding="SAME",    \
                                        activation_fn=tf.nn.relu, normalizer_fn=tf.contrib.layers.batch_norm, \
                                        weights_initializer=initializer,scope="g_conv1")

        conv1 = tf.reshape(conv1, shape=[batch_size,14,14,64])

        conv2 = tf.contrib.layers.conv2d(conv1, num_outputs=4*32, kernel_size=5, stride=1, padding="SAME", \
                                        activation_fn=tf.nn.relu,normalizer_fn=tf.contrib.layers.batch_norm, \
                                        weights_initializer=initializer,scope="g_conv2")

        conv2 = tf.reshape(conv2, shape=[batch_size,28,28,32])
        
        conv3 = tf.contrib.layers.conv2d(conv2, num_outputs=1, kernel_size=5, stride=1, padding="SAME", \
                                        activation_fn=tf.nn.tanh,scope="g_conv3")

        return conv3


def discriminator(tensor,reuse=False):
    
    with tf.variable_scope("discriminator"):

        conv1 = tf.contrib.layers.conv2d(inputs=tensor, num_outputs=32, kernel_size=5, stride=2, padding="SAME", \
                                        reuse=reuse, activation_fn=lrelu,weights_initializer=initializer,scope="d_conv1")

        conv2 = tf.contrib.layers.conv2d(inputs=conv1, num_outputs=64, kernel_size=5, stride=2, padding="SAME", \
                                        reuse=reuse, activation_fn=lrelu,normalizer_fn=tf.contrib.layers.batch_norm,\
                                        weights_initializer=initializer,scope="d_conv2")

        fc1 = tf.reshape(conv2, shape=[batch_size, 7*7*64])

        fc1 = tf.contrib.layers.fully_connected(inputs=fc1, num_outputs=512,reuse=reuse, activation_fn=lrelu, \
                                                normalizer_fn=tf.contrib.layers.batch_norm, \
                                                weights_initializer=initializer,scope="d_fc1")

        fc2 = tf.contrib.layers.fully_connected(inputs=fc1, num_outputs=1, reuse=reuse, activation_fn=tf.nn.sigmoid,\
                                                weights_initializer=initializer,scope="d_fc2")

        return fc2


g_out = generator(z_in)
d_out_fake = discriminator(g_out)
d_out_real = discriminator(x_image,reuse=True)

# loss & optimizer

disc_loss = tf.reduce_sum(tf.square(d_out_real-1) + tf.square(d_out_fake))/2
gen_loss = tf.reduce_sum(tf.square(d_out_fake-1))/2

tvars = tf.trainable_variables() 

gen_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="generator") 
dis_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="discriminator") 

d_optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate)
g_optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate)

d_grads = d_optimizer.compute_gradients(disc_loss,dis_variables) #Only update the weights for the discriminator network.
g_grads = g_optimizer.compute_gradients(gen_loss,gen_variables) #Only update the weights for the generator network.

update_D = d_optimizer.apply_gradients(d_grads)
update_G = g_optimizer.apply_gradients(g_grads)


init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    #print(sess.run(gen_variables))

    for i in range(epoch):
        batch = mnist.train.next_batch(batch_size)
        z_input = np.random.uniform(0,1.0,size=[batch_size,100]).astype(np.float32)

        _, d_loss = sess.run([update_D,disc_loss],feed_dict={x: batch[0], z_in: z_input})
        
        for j in range(10):
            _, g_loss = sess.run([update_G,gen_loss],feed_dict={z_in: z_input})

        print("i: {} / d_loss: {} / g_loss: {}".format(i,np.sum(d_loss)/batch_size, np.sum(g_loss)/batch_size))

        if i % 10 == 0:

            gen_o = sess.run(g_out,feed_dict={z_in: z_input})
            #result = plt.imshow(gen_o[0][:, :, 0], cmap="gray")
            plt.imsave("{}.png".format(i),gen_o[0][:, :, 0], cmap="gray")
