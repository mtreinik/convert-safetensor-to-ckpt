# convert-safetensors-to-ckpt

A simple utility that converts a `.safetensors` file into a `.ckpt` file (TensorFlow checkpoint)

Some old tools may not support the newer `.safetensor` file format. This utility allows converting data to the old `.ckpt` format.

## Usage

./convert_safetensor_to_ckpt.py my-model-file.safetensors my-model-file.ckpt
