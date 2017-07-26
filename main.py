import argparse
import itertools
import numpy as np
import tensorflow as tf
import dcgan

def main(params):
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
  params['num_t_steps'] = params['num_g_steps'] + params['num_d_steps']
  if params['data_set'] == 'mnist':
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
  if params['mode'] == 'train':
    dcgan_est.train(input_fn=lambda: params['input_fn'](params),
      hooks=[logging_hook])
  elif params['mode'] == 'predict':
    pred = dcgan_est.predict(input_fn=lambda: params['input_fn'](params))
    print np.array(list(itertools.islice(pred, 1)))[0]['sampled_images'].shape
    print 'Done'
  else:
    raise Exception('Invalid options to mode')

def parse_arguments():
  parser = argparse.ArgumentParser()
  parser.add_argument('-m', '--mode', type=str, default=None,
    help="mode: train or predict")
  parser.add_argument('-d', '--data_set', type=str, default=None,
    help="mnist or etc.")
  parser.add_argument('-b', '--batch_size', type=int, default=128,
    help='batch size')
  parser.add_argument('-z', '--z_dim', type=int, default=100,
    help='dimension for z vector')
  parser.add_argument('-l', '--learning_rate', type=float, default=0.0002,
    help='learning rate for optimizer')
  parser.add_argument('-i', '--improvements', type=str, default='GAN',
    help='tweaks and tricks to improve GAN training')
  parser.add_argument('-ng', '--num_g_steps', type=int, default=1,
    help='number of g steps to take')
  parser.add_argument('-nd', '--num_d_steps', type=int, default=1,
    help='number of d steps to take')
  parser.add_argument('-cvmax', '--clip_value_min', type=int, default=-0.5,
    help='weights clip value min for WGAN')
  parser.add_argument('-cvmin', '--clip_value_max', type=int, default=0.5,
    help='weights clip value max for WGAN')
  return parser.parse_args()

if __name__ == '__main__':
  params = vars(parse_arguments())
  print params
  main(params)
