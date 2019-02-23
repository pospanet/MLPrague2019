# !!! ML Prague participants notice !!!

I am working on workshop screencast which should be available till end of Feb 2019. In case of any questions, comments, etc. feel free to create new issue here. Thansk

# ML on the EDGE workshop for ML Prague 2019

Main workshop goal is to demonstrate how to automate model deployment to the edge device. So models are artificial and I do care about performance at all.

So if you wanna skip model training part, please download model definition from [here](https://publicsharestorage.blob.core.windows.net/publicshare/MLPrague2019/lego.zip) and you can jump directly to deployment.

# Environment setup

## VS code

As IDE you can use any text editor of your choice. My recommendation would be VS Code. To install please follow [VS Code setup](https://code.visualstudio.com/docs/setup/setup-overview) 

## Python Environment

Most of the sample code is Python. To not to mess with any local Python environment, install Anaconda 3.x (miniconda) environment manager. To do that see [miniconda](https://conda.io/en/latest/miniconda.html)

Once you have conda installed localy, you can create ready-to-go environment using [YAML file](keras_tf_cpu.yaml) by running following command.

`conda env create -f keras_tf_cpu.yaml`

for GPU version you have to have CUDA & cuDDN (see below) drivers/SDK

`conda env create -f keras_tf_gpu.yaml`

## Git client

Code samples can be found in this repo. To interact with, you have to have Git client. Installation guide can be found [here](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git).

## Docker

Last but not least you should have Docker engine to be able to build, deploy and test final project. Docker community edition can be installed from [here](https://hub.docker.com/search/?type=edition&offering=community).

# Dataset

## LEGO bricks
https://www.kaggle.com/joosthazelzet/lego-brick-images

## Stanford Dogs Dataset
http://vision.stanford.edu/aditya86/ImageNetDogs/

# Useful links
[Convert Keras model to TensorFlow](https://github.com/pipidog/keras_to_tensorflow/blob/master/keras_to_tensorflow.py)

[Docker on Azure](https://github.com/pospanet/docker2azure)

[Install IoT Edge for Linux](https://docs.microsoft.com/azure/iot-edge/how-to-install-iot-edge-linux)

[TensorFlow GPU support instalation](https://www.tensorflow.org/install/gpu)