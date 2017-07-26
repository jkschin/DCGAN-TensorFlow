import argparse
import itertools
import numpy as np
import tensorflow as tf
import dcgan

def main():
  config = tf.estimator.RunConfig()
  config._model_dir = '/tmp/dcgan_model'
  config._save_summary_steps = 10
  config._save_checkpoints_secs = None
  config._save_checkpoints_steps = 100
  tensors_to_log = {'g_loss': 'g_loss',
                    'd_loss_fake': 'd_loss_fake',
                    'd_loss_real': 'd_loss_real',
                    'train_gen': 'train_gen',
                    'mod': 'mod',
                    'global_step': 'global_step'}
  logging_hook = tf.train.LoggingTensorHook(
      tensors=tensors_to_log, every_n_iter=1)
  params = {
      'batch_size' : 128,
      'z_dim' : 100,
      'learning_rate' : 0.0002,
      'num_g_steps' : 1,
      'num_d_steps' : 2,
      'num_t_steps' : 3}
  mode = 'mnist'
  if mode == 'mnist':
    import mnist.discriminator
    import mnist.generator
    import mnist.input

    params['generator_fn'] = mnist.generator.generator_fn
    params['discriminator_fn'] = mnist.discriminator.discriminator_fn
    params['input_fn'] = mnist.input.input_fn

  dcgan_est = tf.estimator.Estimator(
      model_fn=dcgan.dcgan_fn,
      config=config,
      params=params)
  # dcgan_est.train(input_fn=lambda: params['input_fn'](params),
  #     hooks=[logging_hook])

  pred = dcgan_est.predict(input_fn=lambda: params['input_fn'](params))
  print np.array(list(itertools.islice(pred, 1)))[0]['sampled_images'].shape
  print 'Done'

def parse_arguments(argv)

if __name__ == '__main__':
  main()
