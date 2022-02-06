import torch
import torch.nn as nn

''' Tuple: (out_channels, kernel_size, stride) '''
''' List: ['B', number_of_repeats] '''
''' B - block of convolutional, convolutional, residual '''
''' S - scale prediction block '''
''' U - upsempling feature map and concatenation with a previous layer '''

config = [
  (32, 3, 1),
  (64, 3, 2),
  ['B', 1],
  (128, 3, 2),
  ['B', 2],
  (256, 3, 2),
  ['B', 8],
  (512, 3, 2),
  ['B', 4], # Darknet-53
  (512, 1, 1),
  (1024, 3, 1),
  "S",
  (256, 1, 1),
  "U",
  (256, 1, 1),
  (512, 3, 1),
  "S",
  (128, 1, 1),
  "U",
  (128, 1, 1),
  (256, 3, 1),
  "S",
  
  ]
