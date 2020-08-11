# Convolutional Hierarchical Attention Network for Query-Focused Video Summarization
code for ***Convolutional Hierarchical Attention Network for Query-Focused Video Summarization***，which is accepted by AAAI 2020 conference.

[arXiv](https://arxiv.org/abs/2002.03740)  [paper](https://doi.org/10.1609/aaai.v34i07.6929) 



## Prerequisites

- Python 3.5
- PyTorch 1.4.0
- matlab.engine (Install MATLAB Engine API for Python [here](https://www.mathworks.com/help/matlab/matlab_external/install-the-matlab-engine-for-python.html))
- gensim
- h5py



## Run

1. download and unzip [glove.840B.300d.zip](http://nlp.stanford.edu/data/glove.840B.300d.zip) and place it at ./data/glove.bin
2. download and unzip [UTC_feature.zip](https://drive.google.com/file/d/1np6d59s27PASZK7yjdnnvkmqT1cPeotO/view?usp=sharing) and place it at ./data/
3. (optional) modify json file in config folder for experimental setting
4. `python main.py`