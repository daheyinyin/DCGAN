# ============================================================================
"""
network config setting, will be used in train.py
"""

from easydict import EasyDict as edict

dcgan_imagenet_cfg = edict({
    'num_classes': 1000,
    'epoch_size': 20,
    'batch_size': 128,
    'latent_size': 100,
    'feature_size': 64,
    'channel_size': 3,
    'image_height': 32,
    'image_width': 32,
    'learning_rate': 0.0002,
    'beta1': 0.5
})

dcgan_cifar10_cfg = edict({
    'num_classes': 10,
    'ds_length': 60000,
    'batch_size': 100,
    'latent_size': 100,
    'feature_size': 64,
    'channel_size': 3,
    'image_height': 32,
    'image_width': 32,
    'learning_rate': 0.0002,
    'beta1': 0.5
})
