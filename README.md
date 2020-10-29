# RUBi : Reducing Unimodal Biases for Visual Question Answering

This is the code for the NeurIPS 2019 article available here: https://arxiv.org/abs/1906.10169.

This paper was written by [Rémi Cadene](http://www.remicadene.com/), [Corentin Dancette](https://cdancette.fr), Hedi Ben Younes, [Matthieu Cord](http://webia.lip6.fr/~cord/) and [Devi Parikh](https://www.cc.gatech.edu/~parikh/).


**RUBi** is a learning strategy to reduce biases in VQA models. 
It relies on a question-only branch plugged at the end of a VQA model. 


<p align="center">
    <img src="https://github.com/cdancette/rubi.bootstrap.pytorch/blob/master/assets/model_classic.png?raw=true" width="300px" style="margin-right:50px;"/>    
    <img src="https://github.com/cdancette/rubi.bootstrap.pytorch/blob/master/assets/model_rubi.png?raw=true" width="300px" style="margin-left: 50px";/>
</p>

#### Summary

* [Installation](#installation)
    * [As a standalone project](#1-as-standalone-project)
    * [As a python library](#1-as-a-python-library)
    * [Download datasets](#3-download-datasets)
* [Quick start](#quick-start)
    * [Train a model](#train-a-model)
    * [Evaluate a model](#evaluate-a-model)
* [Reproduce results](#reproduce-results)
    * [VQACP2](#vqa-CP-v2-dataset)
    * [VQA2](#vqa-v2-dataset)
* [Useful commands](#useful-commands)
* [Authors](#authors)
* [Acknowledgment](#acknowledgment)


## Installation

We don't provide support for python 2. We advise you to install python 3 with [Anaconda](https://www.continuum.io/downloads). Then, you can create an environment.

### 1. As standalone project

```
conda create --name rubi python=3.7
source activate rubi
git clone --recursive https://github.com/cdancette/rubi.bootstrap.pytorch.git
cd rubi.bootstrap.pytorch
pip install -r requirements.txt
```

### (1. As a python library)

To install the library 
```
git clone https://github.com/cdancette/rubi.bootstrap.pytorch.git
python setup.py install
```

Then by importing the `rubi` python module, you can access datasets and models in a simple way.

```python
from rubi.models.networks.rubi import RUBiNet
```


**Note:** This repo is built on top of [block.bootstrap.pytorch](https://github.com/Cadene/block.bootstrap.pytorch). We import VQA2, TDIUC, VGenome from this library.

### 2. Download datasets

Download annotations, images and features for VQA experiments:
```
bash rubi/datasets/scripts/download_vqa2.sh
bash rubi/datasets/scripts/download_vqacp2.sh
```


## Quick start

### The RUBi model

The main model is RUBi. 

```python
from rubi.models.networks.rubi import RUBiNet
```

RUBi takes as input another VQA model, adds a question branch around it. The question predictions are merged with the original predictions.
RUBi returns the new predictions that are used to train the VQA model.

For an example base model, you can check the [baseline model](https://github.com/cdancette/rubi.pytorch/blob/master/rubi/models/networks/baseline_net.py). The model must return the raw predictions (before softmax) in a dictionnary, with the key `logits`.


### Train a model

The [boostrap/run.py](https://github.com/Cadene/bootstrap.pytorch/blob/master/bootstrap/run.py) file load the options contained in a yaml file, create the corresponding experiment directory and start the training procedure. For instance, you can train our best model on VQA2 by running:
```
python -m bootstrap.run -o rubi/options/vqacp2/rubi.yaml
```
Then, several files are going to be created in `logs/vqa2/rubi`:
- [options.yaml](https://github.com/Cadene/block.bootstrap.pytorch/blob/master/assets/logs/vrd/block/options.yaml) (copy of options)
- [logs.txt](https://github.com/Cadene/block.bootstrap.pytorch/blob/master/assets/logs/vrd/block/logs.txt) (history of print)
- [logs.json](https://github.com/Cadene/block.bootstrap.pytorch/blob/master/assets/logs/vrd/block/logs.json) (batchs and epochs statistics)
- [view.html](http://htmlpreview.github.io/?https://raw.githubusercontent.com/Cadene/block.bootstrap.pytorch/master/assets/logs/vrd/block/view.html?token=AEdvLlDSYaSn3Hsr7gO5sDBxeyuKNQhEks5cTF6-wA%3D%3D) (learning curves)
- ckpt_last_engine.pth.tar (checkpoints of last epoch)
- ckpt_last_model.pth.tar
- ckpt_last_optimizer.pth.tar
- ckpt_best_eval_epoch.accuracy_top1_engine.pth.tar (checkpoints of best epoch)
- ckpt_best_eval_epoch.accuracy_top1_model.pth.tar
- ckpt_best_eval_epoch.accuracy_top1_optimizer.pth.tar

Many options are available in the [options directory](https://github.com/cdancette/rubi.bootstrap.pytorch/blob/master/rubi/options).

### Evaluate a model

There is no testing set on VQA-CP v2, our main dataset. The evaluation is done on the validation set.

For a model trained on VQA v2, you can evaluate your model on the testing set. In this example, [boostrap/run.py](https://github.com/Cadene/bootstrap.pytorch/blob/master/bootstrap/run.py) load the options from your experiment directory, resume the best checkpoint on the validation set and start an evaluation on the testing set instead of the validation set while skipping the training set (train_split is empty). Thanks to `--misc.logs_name`, the logs will be written in the new `logs_predicate.txt` and `logs_predicate.json` files, instead of being appended to the `logs.txt` and `logs.json` files.
```
python -m bootstrap.run \
-o logs/vqa2/rubi/baseline.yaml \
--exp.resume best_accuracy_top1 \
--dataset.train_split \
--dataset.eval_split test \
--misc.logs_name test
```


## Reproduce results

### VQA-CP v2 dataset

Use this simple setup to reproduce our results on the valset of VQA-CP v2.

Baseline: 

```bash
python -m bootstrap.run \
-o rubi/options/vqacp2/baseline.yaml \
--exp.dir logs/vqacp2/baseline
```

RUBi : 

```bash
python -m bootstrap.run \
-o rubi/options/vqacp2/rubi.yaml \
--exp.dir logs/vqacp2/rubi
```

#### Compare experiments on valset

You can compare experiments by displaying their best metrics on the valset.

```
python -m rubi.compare_vqacp2_rubi -d logs/vqacp2/rubi logs/vqacp2/baseline
```

### VQA v2 dataset

Baseline: 

```bash
python -m bootstrap.run \
-o rubi/options/vqa2/baseline.yaml \
--exp.dir logs/vqa2/baseline
```

RUBi : 

```bash
python -m bootstrap.run \
-o rubi/options/vqa2/rubi.yaml \
--exp.dir logs/vqa2/rubi
```


You can compare experiments by displaying their best metrics on the valset.

```
python -m rubi.compare_vqa2_rubi_val -d logs/vqa2/rubi logs/vqa2/baseline
```

#### Evaluation on test set

```bash
python -m bootstrap.run \
-o logs/vqa2/rubi/options.yaml \
--exp.resume best_eval_epoch.accuracy_top1 \
--dataset.train_split '' \
--dataset.eval_split test \
--misc.logs_name test
```

## Weights of best model


The weights for the model trained on VQA-CP v2 can be downloaded here : http://webia.lip6.fr/~cadene/rubi/ckpt_last_model.pth.tar

To use it : 
* Run this command once to create the experiment folder. Cancel it when the training starts

```bash
python -m bootstrap.run \
-o rubi/options/vqacp2/rubi.yaml \
--exp.dir logs/vqacp2/rubi
```

* Move the downloaded file to the experiment folder, and use the flag `--exp.resume last` to use this checkpoint : 

```bash
python -m bootstrap.run \
-o logs/vqacp2/rubi/options.yaml \
--exp.resume last
```


## Useful commands

### Use tensorboard instead of plotly

Instead of creating a `view.html` file, a tensorboard file will be created:
```
python -m bootstrap.run -o rubi/options/vqacp2/rubi.yaml \
--view.name tensorboard
```

```
tensorboard --logdir=logs/vqa2
```

You can use plotly and tensorboard at the same time by updating the yaml file like [this one](https://github.com/Cadene/bootstrap.pytorch/blob/master/bootstrap/options/mnist_plotly_tensorboard.yaml#L38).


### Use a specific GPU

For a specific experiment:
```
CUDA_VISIBLE_DEVICES=0 python -m boostrap.run -o rubi/options/vqacp2/rubi.yaml
```

For the current terminal session:
```
export CUDA_VISIBLE_DEVICES=0
```

### Overwrite an option

The boostrap.pytorch framework makes it easy to overwrite a hyperparameter. In this example, we run an experiment with a non-default learning rate. Thus, I also overwrite the experiment directory path:
```
python -m bootstrap.run -o rubi/options/vqacp2/rubi.yaml \
--optimizer.lr 0.0003 \
--exp.dir logs/vqacp2/rubi_lr,0.0003
```

### Resume training

If a problem occurs, it is easy to resume the last epoch by specifying the options file from the experiment directory while overwritting the `exp.resume` option (default is None):
```
python -m bootstrap.run -o logs/vqacp2/rubi/options.yaml \
--exp.resume last
```


## Authors

This code was made available by [Corentin Dancette](https://cdancette.fr) and [Rémi Cadene](http://www.remicadene.com/)

## Acknowledgment

Special thanks to the authors of [VQA2](TODO), [TDIUC](TODO), [VisualGenome](TODO) and [VQACP2](TODO), the datasets used in this research project.
