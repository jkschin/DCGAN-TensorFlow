import tensorflow as tf
from helpers import leaky_relu as leaky_relu

def conv_bn_nonlinearity(inputs, filters, kernel_size, strides, padding,
    activation, training):
  conv = tf.layers.conv2d(
      inputs=inputs,
      filters=filters,
      kernel_size=kernel_size,
      strides=strides,
      padding=padding,
      activation=None,
      kernel_regularizer=tf.contrib.layers.l2_regularizer(1.0),
      bias_regularizer=tf.contrib.layers.l2_regularizer(1.0))
  bn = tf.layers.batch_normalization(conv, training=training)
  a = activation(bn)
  return a

def discriminator_fn(features, labels, mode, reuse):
  with tf.variable_scope('discriminator', reuse=reuse):
    training = mode == tf.estimator.ModeKeys.TRAIN
    layers = [features]
    layers.append(conv_bn_nonlinearity(
      layers[-1], 128, [5, 5], [2, 2], 'same', leaky_relu, training))
    layers.append(conv_bn_nonlinearity(
      layers[-1], 256, [5, 5], [2, 2], 'same', leaky_relu, training))

    # Note that this is explicitly None because
    # tf.nn.sigmoid.cross_entropy_with_logits is used in the train step.
    layers.append(tf.reshape(layers[-1], [-1, 7*7*256]))
    layers.append(tf.layers.dense(layers[-1], 1, activation=None))
    for l in layers:
      print l
    return layers[-1]
