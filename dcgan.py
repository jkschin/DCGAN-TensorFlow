# import cv2
import numpy as np
import os
import tensorflow as tf
from tensorflow.contrib import learn

tf.logging.set_verbosity(tf.logging.INFO)

# TODO: efficient mean and variance functions

def leaky_relu(x):
    negative_part = tf.nn.relu(-x)
    x = tf.nn.relu(x)
    x -= tf.constant(0.01, tf.float32) * negative_part
    return x

class DCGAN:
  def __init__(self, features, labels, batch_size, z_dim, learning_rate,
      num_g_steps, num_d_steps):
    self.features = features
    self.labels = labels
    self.batch_size = batch_size
    self.z_dim = z_dim
    self.learning_rate = learning_rate
    self.num_g_steps = num_g_steps
    self.num_d_steps = num_d_steps
    self.num_t_steps = num_g_steps + num_d_steps
    self.G = Generator()
    self.D = Discriminator()

  def input_fn(self):
    mnist = learn.datasets.load_dataset("mnist")
    # Returns np.array
    real_images = mnist.train.images.reshape((55000, 28, 28, 1))
    real_image = tf.train.slice_input_producer([tf.constant(real_images)])
    z = tf.random_uniform([self.z_dim], minval=-1, maxval=1)

    min_after_dequeue = 10000
    capacity = min_after_dequeue + 3 * self.batch_size
    features = tf.train.batch(
        { 'real_images': real_image,
          'z': z},
        batch_size=self.batch_size,
        capacity=32)

    # NOTE for some reason there's a need to squeeze here when it shouldn't be
    # necessary.
    features['real_images'] = tf.squeeze(features['real_images'], axis=1)
    labels = None
    return features, labels

  def train(self):
    def dcgan_fn(features, labels, mode):
      global_step_tensor = tf.Variable(0, trainable=False, dtype=tf.int64,
          name='global_step_tensor')
      real_images = features['real_images']
      z = features['z']
      sampled_images = self.G.generator_fn(z, None, mode, False)
      D_logits_real = self.D.discriminator_fn(real_images, None, mode, False)
      D_logits_fake = self.D.discriminator_fn(sampled_images, None, mode, True)
      print D_logits_real
      print D_logits_fake
      tf.identity(D_logits_real, name='d_logits_real')
      tf.identity(D_logits_fake, name='d_logits_fake')

      g_vars = tf.get_collection(
          tf.GraphKeys.TRAINABLE_VARIABLES,
          scope='generator')
      d_vars = tf.get_collection(
          tf.GraphKeys.TRAINABLE_VARIABLES,
          scope='discriminator')

      if mode != tf.estimator.ModeKeys.PREDICT:
        g_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
              logits=D_logits_fake,
              labels=tf.ones_like(D_logits_fake)),
            name='g_loss')
        d_loss_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
              logits=D_logits_real,
              labels=tf.ones_like(D_logits_real)),
            name='d_loss_real')
        d_loss_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
              logits=D_logits_fake,
              labels=tf.zeros_like(D_logits_fake)),
            name='d_loss_fake')
        d_loss = d_loss_real + d_loss_fake

      if mode == tf.estimator.ModeKeys.TRAIN:
        mod = tf.mod(global_step_tensor, self.num_t_steps, name='mod')
        train_gen = tf.less(mod, self.num_g_steps, name='train_gen')
        g_optim = tf.train.AdamOptimizer(self.learning_rate).minimize(
            g_loss,
            var_list=g_vars)
        d_optim = tf.train.AdamOptimizer(self.learning_rate).minimize(
            d_loss,
            var_list=d_vars)
        optim_op = tf.cond(train_gen, lambda: g_optim, lambda: d_optim,
            name='optim_op')
        loss = tf.cond(train_gen, lambda: g_loss, lambda: d_loss, name='loss')
        with tf.control_dependencies([mod, optim_op]):
          global_inc_op = tf.assign_add(global_step_tensor, 1, use_locking=True)
        train_op = tf.group(optim_op, global_inc_op)

      predictions = {
          'sampled_images': sampled_images,
          'probabilities': D_logits_fake
          }

      return tf.estimator.EstimatorSpec(
          mode=mode, predictions=predictions, loss=loss, train_op=train_op)
    config = tf.estimator.RunConfig()
    config._model_dir = '/tmp/dcgan_model'
    config._save_summary_steps = 1
    config._save_checkpoints_secs = None
    config._save_checkpoints_steps = 100
    dcgan = tf.estimator.Estimator(
        model_fn=dcgan_fn,
        config=config)
    tensors_to_log = {'d_logits_fake': 'd_logits_fake',
                      'd_logits_real': 'd_logits_real',
                      'g_loss': 'g_loss',
                      'd_loss_fake': 'd_loss_fake',
                      'd_loss_real': 'd_loss_real',
                      'train_gen': 'train_gen',
                      'mod': 'mod',
                      'global_step': 'global_step_tensor'}
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=1)
    dcgan.train(input_fn=self.input_fn, hooks=[logging_hook])


class Generator:
  def __init__(self):
    pass

  def generator_fn(self, features, labels, mode, reuse):
    with tf.variable_scope('generator', reuse=reuse):
      training = mode == tf.estimator.ModeKeys.TRAIN
      layers = [features]
      layers.append(tf.layers.dense(layers[-1], 4*4*1024, activation=leaky_relu))
      layers.append(tf.layers.batch_normalization(layers[-1], training=training))
      layers.append(tf.reshape(layers[-1], [-1, 4, 4, 1024]))
      layers.append(tf.layers.conv2d_transpose(
        inputs=layers[-1],
        filters=512,
        kernel_size=[5, 5],
        strides=[2, 2],
        padding='same',
        activation=leaky_relu))
      layers.append(tf.layers.batch_normalization(layers[-1], training=training))
      layers.append(tf.layers.conv2d_transpose(
        inputs=layers[-1],
        filters=256,
        kernel_size=[5, 5],
        strides=[2, 2],
        padding='same',
        activation=leaky_relu))
      layers.append(tf.layers.batch_normalization(layers[-1], training=training))
      # layers.append(tf.layers.conv2d_transpose(
      #   inputs=layers[-1],
      #   filters=128,
      #   kernel_size=[5, 5],
      #   strides=[2, 2],
      #   padding='same',
      #   activation=leaky_relu))
      # layers.append(tf.layers.batch_normalization(layers[-1], training=training))
      layers.append(tf.layers.conv2d_transpose(
        inputs=layers[-1],
        filters=1,
        kernel_size=[5, 5],
        strides=[2, 2],
        padding='same',
        activation=tf.nn.tanh))
      return layers[-1]

class Discriminator:
  def __init__(self):
    pass

  def discriminator_fn(self, features, labels, mode, reuse):
    with tf.variable_scope('discriminator', reuse=reuse):
      training = mode == tf.estimator.ModeKeys.TRAIN
      layers = [features]
      layers.append(tf.layers.conv2d(
        layers[-1],
        filters=64,
        kernel_size=[5, 5],
        strides=[2, 2],
        padding='same',
        activation=leaky_relu))
      layers.append(tf.layers.batch_normalization(layers[-1], training=training))
      layers.append(tf.layers.conv2d(
        layers[-1],
        filters=128,
        kernel_size=[5, 5],
        strides=[2, 2],
        padding='same',
        activation=leaky_relu))
      layers.append(tf.layers.batch_normalization(layers[-1], training=training))
      layers.append(tf.layers.conv2d(
        layers[-1],
        filters=256,
        kernel_size=[5, 5],
        strides=[2, 2],
        padding='same',
        activation=leaky_relu))
      layers.append(tf.layers.batch_normalization(layers[-1], training=training))
      # layers.append(tf.layers.conv2d(
      #   layers[-1],
      #   filters=512,
      #   kernel_size=[5, 5],
      #   strides=[2, 2],
      #   padding='same',
      #   activation=leaky_relu))
      # layers.append(tf.layers.batch_normalization(layers[-1], training=training))

      # Note that this is explicitly None because
      # tf.nn.sigmoid.cross_entropy_with_logits is used in the train step.
      layers.append(tf.reshape(layers[-1], [-1, 4*4*256]))
      layers.append(tf.layers.dense(layers[-1], 1, activation=None))
      for l in layers:
        print l
      return layers[-1]

class Input:
  # mean and stddev computed included data from test set, for convenience
  # BGR Mean [  88.26726517  103.51948437  133.39556285]
  # BGR Var [ 61.1038604   63.8995506   73.49262324]
  # YCRCB Mean [ 110.71413264  144.19812304  115.33745339]
  # YCRCB Var [ 65.01511419  14.50090058  12.46935316]
  def __init__(self, image_directory):
    self.image_directory = image_directory
    self.mean = tf.constant([88.26726517, 103.51948437, 133.39556285])
    self.stddev = tf.constant([61.1038604, 63.8995506, 73.49262324])

  def normalize_and_crop(self, image):
    image = tf.image.central_crop(image, 0.5)
    image = (image - self.mean) / self.std
    return image

  def data_augmentation(self, image):
    image = tf.image.random_flip_left_right(image, seed=1)
    image = tf.image.random_brightness(image, 0.5)
    image = tf.image.random_contrast(image, 0.5, 1)
    return image

  def read_images(self, input_queue):
    reader = tf.WholeFileReader()
    key, value = reader.read(input_queue)
    image = tf.image.decode_image(value, channels=3)
    return image

  def input_fn(self, image_names, num_epochs):
    filename_queue = tf.train.string_input_producer(
        [image_names],
        num_epochs=num_epochs,
        shuffle=True)
    image = self.read_images(filename_queue)
    image = self.normalize_and_crop(image)
    image = self.data_augmentation(image)
    min_after_dequeue = 10000
    capacity = min_after_dequeue + 3 * batch_size
    images = tf.train.shuffle_batch(
        image,
        batch_size=batch_size,
        capacity=capacity,
        min_after_dequeue=min_after_dequeue)
    return images, None


def main():
  features = None
  labels = None
  batch_size = 100
  z_dim = 100
  learning_rate = 0.01
  num_g_steps = 1
  num_d_steps = 5
  dcgan = DCGAN(features, labels, batch_size, z_dim, learning_rate,
      num_g_steps, num_d_steps)
  dcgan.train()
  # obj = Preprocess('/media/jkschin/WD2TB/data/CelebA/img_align_celeba_64x64/',
  #     (64, 64, 3))
  # obj.get_mean()


if __name__ == '__main__':
  main()

# class Preprocess:
#   def __init__(self, image_directory, shape):
#     self.image_directory = image_directory
#     # assumes only images in dir
#     self.num_images = len(os.listdir(self.image_directory))
#     self.h, self.w, self.c = shape

#   def get_mean(self):
#     pixel_count = float(self.num_images * self.h * self.w * self.c)
#     pixel_sum = np.zeros((3), dtype=np.int64)
#     i = 0
#     for image_name in os.listdir(self.image_directory):
#       img = cv2.imread(os.path.join(self.image_directory, image_name))
#       img = img.reshape((self.h*self.w, 3)).astype(np.int64)
#       img = np.sum(img, axis=0)
#       pixel_sum += img
#       i += 1
#       print i
#       # print pixel_sum
#     print pixel_sum / pixel_count

# class Preprocess:
#   def __init__(self, image_directory):
#     self.image_directory = image_directory

#     self.output_directory = image_directory.split('/')
#     self.output_directory[-1] = 'img_align_celeba_64x64'
#     self.output_directory = '/'.join(self.output_directory)

#     if not os.path.exists(self.output_directory):
#       os.makedirs(self.output_directory)

#     image_names = map(lambda x: os.path.join(image_directory, x),
#         os.listdir(image_directory))
#     filename_queue = tf.train.string_input_producer(image_names, num_epochs=1)
#     reader = tf.WholeFileReader()
#     key, value = reader.read(filename_queue)
#     image = tf.image.decode_image(value, channels=3)
#     image = tf.image.resize_image_with_crop_or_pad(image, 64, 64)
#     image = tf.image.encode_png(image)
#     sess = tf.Session()
#     sess.run(
#         tf.group(
#           tf.global_variables_initializer(),
#           tf.local_variables_initializer()))
#     coord = tf.train.Coordinator()
#     threads = tf.train.start_queue_runners(sess=sess, coord=coord)
#     i = 0
#     while True:
#       try:
#         key_val, image_val = sess.run([key, image])
#         key_val = key_val.split('/')[-1][:-4] + '.png'
#         key_val = os.path.join(self.output_directory, key_val)
#         f = open(key_val, 'wb')
#         f.write(image_val)
#         f.close()
#         print i
#         print key_val
#       except tf.errors.OutOfRangeError:
#         print 'OutOfRange'
#         break
#       i += 1
