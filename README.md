# Convolutional Hierarchical Attention Network for Query-Focused Video Summarization
code for ***Convolutional Hierarchical Attention Network for Query-Focused Video Summarization***，which is accepted by AAAI 2020 conference.

[arXiv](https://arxiv.org/abs/2002.03740)  [paper](https://doi.org/10.1609/aaai.v34i07.6929) 



## Prerequisites

- Python3
- PyTorch 1.5+
- matlab.engine (Install MATLAB Engine API for Python [here](https://www.mathworks.com/help/matlab/matlab_external/install-the-matlab-engine-for-python.html))



## Run

extract resent and c3d feature of 4 videos mentioned in paper and put them in ./data/features folder with h5 format

`python ./data/preprocess.py`

`python ./data/query_feature.py`

modify config/config.py for experimental setting

`python main.py`


