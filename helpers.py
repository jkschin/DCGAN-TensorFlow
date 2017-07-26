import tensorflow as tf

def leaky_relu(x):
  return tf.maximum(0.1*x, x)
