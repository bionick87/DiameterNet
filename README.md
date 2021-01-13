


# Real-time diameter of the fetal aorta from ultrasound

![Alt text](img/model.gif?raw=true "model")


## Motivation

The automatic analysis of ultrasound sequences can substantially improve the efficiency of clinical diagnosis. This article presents an attempt to automate the challenging task of measuring the vascular diameter of the fetal abdominal aorta from ultrasound images. We propose a neural network architecture consisting of three blocks: a convolutional neural network (CNN) for the extraction of imaging features, a convolution gated recurrent unit (C-GRU) for exploiting the temporal redundancy of the signal, and a regularized loss function, called CyclicLoss, to impose our prior knowledge about the periodicity of the observed signal. The solution is investigated with a cohort of 25 ultrasound sequences acquired during the third-trimester pregnancy check, and with 1000 synthetic sequences. In the extraction of features, it is shown that a shallow CNN outperforms two other deep CNNs with both the real and synthetic cohorts, suggesting that echocardiographic features are optimally captured by a reduced number of CNN layers. The proposed architecture, working with the shallow CNN, reaches an accuracy substantially superior to previously reported methods, providing an average reduction of the mean squared error from 0.31 (state-of-the-art) to 0.09 mm2, and a relative error reduction from 8.1 to 5.3%. The mean execution speed of the proposed approach of 289 frames per second makes it suitable for real-time clinical use.


# Installing the dependencies

Before getting started with DiameterNet, it's important to have a working environment with all dependencies satisfied. For this, we recommend using the Anaconda distribution of Python 3.5. 


```bash

cd /tmp
curl -O https://repo.anaconda.com/archive/Anaconda3-2019.03-Linux-x86_64.sh
bash Anaconda3-2019.03-Linux-x86_64.sh

```

So pytorch must be installed, please make sure that cuDNN is installed correctly (https://developer.nvidia.com/cudnn).

```bash
conda install pytorch torchvision cudatoolkit=9.2 -c pytorch

```

Then install the following libraries.

```bash
pip install torchvision
pip install opencv-python
pip install matplotlib
pip install progressbar
pip install pytest-shutil
pip install matplotlib
```

# Use of the code

Use the following command to start training. 

```bash
CUDA_VISIBLE_DEVICES=0 python main.py --typeExtractor alexnet --typeRecurrent unidir --typeCriterion MSECyclic --typeMeasurement Iamt --typeDataset Real
```
In main.py you find a list of parameters and paths to configure (i. learning, frame).

Suggested GPU hardware is GeForce RTX 2080 Ti.

To use code inference:


```bash
CUDA_VISIBLE_DEVICES=0 python inference.py
```

# Citations

If you have used the DiameterNet code, please also cite:

Savioli, N., Grisan, E., Visentin, S. et al. Real-time diameter of the fetal aorta from ultrasound. Neural Comput & Applic 32, 6735–6744 (2020). https://doi.org/10.1007/s00521-019-04646-3



# License

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
