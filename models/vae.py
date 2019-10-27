import tensorflow as tf


def reset_graph():
  if 'sess' in globals() and sess:
    sess.close()
  tf.reset_default_graph()


class ConvVAE(object):
  def __init__(self, name, z_size=32):
    self.name = name
    self.z_size = z_size

    # initialized
    with tf.variable_scope(name):
      self.scope = tf.get_variable_scope().name

  # more like a call function
  def build_encoder(self, x, reuse=False):
    # it should be called in the scope where the instance is created.
    with tf.variable_scope(self.name):
      with tf.variable_scope("encoder", reuse=reuse):
        batch_size = tf.shape(x)[0]
        h = tf.layers.conv2d(x, 32, 4, strides=2, activation=tf.nn.relu,
                             name="enc_conv1")
        h = tf.layers.conv2d(h, 64, 4, strides=2, activation=tf.nn.relu,
                             name="enc_conv2")
        h = tf.layers.conv2d(h, 128, 4, strides=2,
                             activation=tf.nn.relu, name="enc_conv3")
        h = tf.layers.conv2d(h, 256, 4, strides=2,
                             activation=tf.nn.relu, name="enc_conv4")
        h = tf.reshape(h, [-1, 2 * 2 * 256])

        # VAE
        self.mean = tf.layers.dense(h, self.z_size, name="enc_fc_mean")
        self.logvar = tf.layers.dense(h, self.z_size,
                                      name="enc_fc_log_var")
        sigma = tf.exp(self.logvar / 2.0)
        epsilon = tf.random_normal([batch_size, self.z_size])
        z = self.mean + sigma * epsilon
        return self.mean, self.logvar, z

  def build_decoder(self, z, reuse=False):
    with tf.variable_scope(self.name):
      with tf.variable_scope("decoder", reuse=reuse):
        h = tf.layers.dense(z, 4 * 256, name="dec_fc")
        h = tf.reshape(h, [-1, 1, 1, 4 * 256])
        h = tf.layers.conv2d_transpose(h, 128, 5, strides=2,
                                       activation=tf.nn.relu,
                                       name="dec_deconv1")
        h = tf.layers.conv2d_transpose(h, 64, 5, strides=2,
                                       activation=tf.nn.relu,
                                       name="dec_deconv2")
        h = tf.layers.conv2d_transpose(h, 32, 6, strides=2,
                                       activation=tf.nn.relu,
                                       name="dec_deconv3")
        y = tf.layers.conv2d_transpose(h, 1, 6, strides=2,
                                       activation=tf.nn.sigmoid,
                                       name="dec_deconv4")
        return y

  def get_variables(self):
    return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)

  def get_enc_variables(self):
    return [v for v in self.get_variables() if 'enc' in v.name]

  def get_dec_variables(self):
    return [v for v in self.get_variables() if 'dec' in v.name]

  def get_fc_variables(self):
    return [v for v in self.get_variables() if 'fc' in v.name]

  def get_trainable_variables(self):
    return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)
