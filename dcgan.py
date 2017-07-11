import cv2
import numpy as np
import os
import tensorflow as tf
from tensorflow.contrib import learn
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib

# TODO: efficient mean and variance functions

def leaky_relu(x):
    negative_part = tf.nn.relu(-x)
    x = tf.nn.relu(x)
    x -= tf.constant(0.01, tf.float32) * negative_part
    return x

class DCGAN:
  def __init__(self, features, labels, mode):
    with tf.variable_scope('generator'):
      self.G = generator()
    with tf.variable_scope('discriminator'):
      self.D = discriminator()

  def (self):

class Generator:
  def __init__(self):
    pass

  def generator_fn(self, features, labels, mode):
    training = mode == learn.ModeKeys.TRAIN
    layers = [features]
    layers.append(tf.layers.dense(layers[-1], 4*4*1024, activation=leaky_relu))
    layers.append(tf.layers.batch_normalization(layers[-1], training=training))
    layers.append(tf.reshape(layers[-1], [-1, 4, 4, 1024]))
    layers.append(tf.layers.conv2d_transpose(
      inputs=layers[-1],
      filters=512,
      kernel_size=[5, 5],
      strides=[2, 2],
      padding='valid',
      activation=leaky_relu))
    layers.append(tf.layers.batch_normalization(layers[-1], training=training))
    layers.append(tf.layers.conv2d_transpose(
      inputs=layers[-1],
      filters=256,
      kernel_size=[5, 5],
      strides=[2, 2],
      padding='valid',
      activation=leaky_relu))
    layers.append(tf.layers.batch_normalization(layers[-1], training=training))
    layers.append(tf.layers.conv2d_transpose(
      inputs=layers[-1],
      filters=128,
      kernel_size=[5, 5],
      strides=[2, 2],
      padding='valid',
      activation=leaky_relu))
    layers.append(tf.layers.batch_normalization(layers[-1], training=training))
    layers.append(tf.layers.conv2d_transpose(
      inputs=layers[-1],
      filters=3,
      kernel_size=[5, 5],
      strides=[2, 2],
      padding='valid',
      activation=tf.nn.tanh))
    return layers[-1]

class Discriminator:
  def __init__(self):
    pass

  def discriminator_fn(features, labels, mode):
    training = mode == learn.ModeKeys.TRAIN
    layers = [features]
    layers.append(tf.layers.conv2d(
      layers[-1],
      filters=64,
      kernel_size=[5, 5],
      strides=[2, 2],
      padding='valid',
      activation=leaky_relu))
    layers.append(tf.layers.batch_normalization(layers[-1], training=training))
    layers.append(tf.layers.conv2d(
      layers[-1],
      filters=128,
      kernel_size=[5, 5],
      strides=[2, 2],
      padding='valid',
      activation=leaky_relu))
    layers.append(tf.layers.batch_normalization(layers[-1], training=training))
    layers.append(tf.layers.conv2d(
      layers[-1],
      filters=256,
      kernel_size=[5, 5],
      strides=[2, 2],
      padding='valid',
      activation=leaky_relu))
    layers.append(tf.layers.batch_normalization(layers[-1], training=training))
    layers.append(tf.layers.conv2d(
      layers[-1],
      filters=512,
      kernel_size=[5, 5],
      strides=[2, 2],
      padding='valid',
      activation=leaky_relu))
    layers.append(tf.layers.batch_normalization(layers[-1], training=training))
    layers.append(tf.layers.dense(layers[-1], 1, activation=tf.nn.sigmoid))
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
    filename_queue = tf.train.slice(
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
  obj = Preprocess('/media/jkschin/WD2TB/data/CelebA/img_align_celeba_64x64/',
      (64, 64, 3))
  obj.get_mean()


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
