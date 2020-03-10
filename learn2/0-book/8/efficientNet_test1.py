# coding =utf-8

import keras_efficientnets


input_size = 224
model = keras_efficientnets.EfficientNetB5( classes=10, include_top=False, weights='imagenet', input_shape=(input_size,input_size,3))
model.summary()

conv_base = keras_efficientnets.EfficientNetB5(
    weights='imagenet',
    include_top=False,
    input_shape=(input_size,input_size,3),
)

conv_base.summary()
#
# block_args_list = (
#     # First number is `input_channels`, second is `output_channels`.
#     keras_efficientnets.BlockArgs(32, 16, kernel_size=3, strides=(1, 1), num_repeat=1, se_ratio=0.25, expand_ratio=1),
#     keras_efficientnets.BlockArgs(16, 24, kernel_size=3, strides=(2, 2), num_repeat=2, se_ratio=0.25, expand_ratio=6),
#     ...
#
# )






