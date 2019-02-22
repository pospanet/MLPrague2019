# ML on the EDGE workshop for ML Prague 2019
Main workshop goal is to demonstrate how to automate model deployment to the edge device. So models are artificial and I do care about performance at all.

# Environment setup

## VS code

As IDE you can use any text editor of your choice. My recommendation would be VS Code. To install please follow [VS Code setup](https://code.visualstudio.com/docs/setup/setup-overview) 

## Python Environment

Most of the sample code is Python. To not to mess with any local Python environment, install Anaconda 3.x (miniconda) environment manager. To do that see [miniconda](https://conda.io/en/latest/miniconda.html)

Once you have conda installed localy, you can create ready-to-go environment using [YAML file](keras_tf_cpu.yaml) by running following command.

`conda env create -f keras_tf_cpu.yaml`

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