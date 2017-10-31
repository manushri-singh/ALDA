# ALDA - ANN Training

## Environment and Installation Instructions
Language: Python 2.7 on an Ubuntu 16.04 machine
Pip Version: 8.1.1
Package (and its version): 
Keras (2.0.8) , Numpy (1.13.3) , Scipy (1.0.0), matplotlib (2.1.0), h5py (2.7.1)
Backend for neural networks: TensorFlow (version: 1.3.0)
Installation:
1.	On an Ubuntu 16.04 machine, install Python, pip and virtualenv
    a.	Update and Upgrade: sudo apt-get update
    b.	Install pip and virtualenv: sudo apt-get install openjdk-8-jdk git python-dev python3-dev python-numpy python3-numpy build-essential python-pip python3-pip python-virtualenv swig python-wheel libcurl3-dev
    c.	Create a virtualenv: virtualenv --system-site-packages -p python ~/keras-tf-venv

2.	Install TensorFlow: 
    a.	Access the virtualenv: source ~/keras-tf-venv/bin/activate 
    b.	Install TensorFlow: pip install --upgrade tensorflow 

3.	Install Keras and its dependencies:
    a.	pip install numpy scipy
    b.	pip install scikit-learn
    c.	pip install pillow
    d.	pip install h5py
    e.	pip install keras

4.	Install matplotlib: pip install matplotlib
5.	In order to run the code, please execute python hw3q5.py in the shell.

Link Referred for Installation: http://deeplearning.lipingyang.org/2017/08/01/install-keras-with-tensorflow-backend/

