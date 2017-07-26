import tensorflow as tf
from tensorflow.contrib import learn

def input_fn(params):
  mnist = learn.datasets.load_dataset("mnist")
  # Returns np.array, range [0, 1.0]
  real_images = mnist.train.images.reshape((55000, 28, 28, 1))
  real_images = (real_images - 0.5) * 2
  real_image = tf.train.slice_input_producer([tf.constant(real_images)])

  min_after_dequeue = 10000
  capacity = min_after_dequeue + 3 * params['batch_size']
  real_images = tf.train.batch(
      real_image,
      batch_size=params['batch_size'],
      capacity=32)
  # https://github.com/soumith/ganhacks, use normal
  zs = tf.random_normal((params['batch_size'], params['z_dim']))
  features = {
      'real_images': real_images,
      'zs': zs}
  labels = None
  return features, labels
