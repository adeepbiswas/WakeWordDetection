# Wake Word Detection using Transformers

The baseline model for this project has been taken from [Keyword Transformer: A Self-Attention Model for Keyword Spotting](https://arxiv.org/abs/2104.00769).


## Setup

### Download Google Speech Commands

There are two versions of the dataset, V1 and V2. To download and extract dataset V2, run:

```shell
wget https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_v0.02.tar.gz
mkdir data2
mv ./speech_commands_v0.02.tar.gz ./data2
cd ./data2
tar -xf ./speech_commands_v0.02.tar.gz
cd ../
```

### Install dependencies

Set up a new virtual environment:

```shell
pip install virtualenv
virtualenv --system-site-packages -p python3 ./venv3
source ./venv3/bin/activate
```

To install dependencies, run

```shell
pip install -r requirements.txt
```

## Model
The Keyword-Transformer model is defined [here](kws_streaming/models/kws_transformer.py). It takes the mel scale spectrogram as input, which has shape 98 x 40 using the default settings, corresponding to the 98 time windows with 40 frequency coefficients.

There are three variants of the Keyword-Transformer model:

* **Time-domain attention**: each time-window is treated as a patch, self-attention is computed between time-windows
* **Frequency-domain attention**: each frequency is treated as a patch self-attention is computed between frequencies
* **Combination of both**: The signal is fed into both a time- and a frequency-domain transformer and the outputs are combined
* **Patch-wise attention**: Similar to the vision transformer, it extracts rectangular patches from the spectrogram, so attention happens both in the time and frequency domain simultaneously.

## Training a model from scratch
To train KWT-3 from scratch on Speech Commands V2, run  

```shell
sh train.sh
```

Please note that the train directory (given by the argument  `--train_dir`) cannot exist prior to start script.

The model-specific arguments for KWT are:

```shell
--num_layers 12 \ #number of sequential transformer encoders
--heads 3 \ #number of attentions heads
--d_model 192 \ #embedding dimension
--mlp_dim 768 \ #mlp-dimension
--dropout1 0. \ #dropout in mlp/multi-head attention blocks
--attention_type 'time' \ #attention type: 'time', 'freq', 'both' or 'patch'
--patch_size '1,40' \ #spectrogram patch_size, if patch attention is used
--prenorm False \ # if False, use postnorm
```

## Training with distillation

We employ hard distillation from a convolutional model (Att-MH-RNN), similar to the approach in [DeIT](https://github.com/facebookresearch/deit).

To train KWT-3 with hard distillation from a pre-trained model, run

```shell
sh distill.sh
```

To perform inference on Google Speech Commands v2 with 12 labels, run

```shell
sh eval.sh
```

Specific Experiments can be found at branches-
1. Mish
2. nonorm_swish
3. norm_change

## Acknowledgement

This repository has been forked from https://github.com/ARM-software/keyword-transformer
