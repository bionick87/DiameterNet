


# DiameterNet: Temporal Convolution Networks for Real-Time Abdominal Fetal Aorta Analysis with Ultrasound

![Alt text](img/model.gif?raw=true "model")


## Motivation

The automatic analysis of ultrasound sequences can substantially improve the efficiency of clinical diagnosis. In this work we present our attempt to automate the challenging task of measuring the vascular diameter of the fetal abdominal aorta from ultrasound images. We propose a neural network architecture consisting of three blocks: a convolutional layer for the extraction of imaging features, a Convolution Gated Recurrent Unit (C-GRU) for enforcing the temporal coherence across video frames and exploiting the temporal redundancy of a signal, and a regularized loss function, called CyclicLoss, to impose our prior knowledge about the periodicity of the observed signal. We present experimental evidence suggesting that the proposed architecture can reach an accuracy substantially superior to previously proposed methods, providing an average reduction of the mean squared error from   0.31mm2  (state-of-art) to   0.09mm2 , and a relative error reduction from 8.1% to 5.3%. The mean execution speed of the proposed approach of 289 frames per second makes it suitable for real time clinical use.


![Alt text](img/model.png?raw=true "model")


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





#Citing Data sets and Code

Sharing data and code is crucial for reproducibility and scientific progress, and should be rewarded. If you are reusing any of the shared data sets, flows or runs/studies, please honor their respective licences and citation requests.

If you have used the DiameterNet code, please also cite:

@inproceedings{10.1007/978-3-030-01421-6_15,
	risfield_0_da = "2018//",
	risfield_1_bt = "Artificial Neural Networks and Machine Learning – ICANN 2018",
	abstract = "The automatic analysis of ultrasound sequences can substantially improve the efficiency of clinical diagnosis. In this work we present our attempt to automate the challenging task of measuring the vascular diameter of the fetal abdominal aorta from ultrasound images. We propose a neural network architecture consisting of three blocks: a convolutional layer for the extraction of imaging features, a Convolution Gated Recurrent Unit (C-GRU) for enforcing the temporal coherence across video frames and exploiting the temporal redundancy of a signal, and a regularized loss function, called CyclicLoss, to impose our prior knowledge about the periodicity of the observed signal. We present experimental evidence suggesting that the proposed architecture can reach an accuracy substantially superior to previously proposed methods, providing an average reduction of the mean squared error from $$0.31\,\mathrm{mm}^2$$(state-of-art) to $$0.09\,\mathrm{mm}^2$$, and a relative error reduction from $$8.1\%$$to $$5.3\%$$. The mean execution speed of the proposed approach of 289 frames per second makes it suitable for real time clinical use.",
	author = "{Savioli Nicoló} and {Visentin Silvia} and {Cosmi Erich} and {Grisan Enrico} and {Lamata Pablo} and {Montana Giovanni}",
	editor = "{Kůrková Věra} and {Manolopoulos Yannis} and {Hammer Barbara} and {Iliadis Lazaros} and {Maglogiannis Ilias}",
	issn = "978-3-030-01421-6",
	location = "Cham",
	pages = "148–157",
	publisher = "Springer International Publishing",
	title = "{Temporal Convolution Networks for Real-Time Abdominal Fetal Aorta Analysis with Ultrasound}",
	year = "2018"
}

we also have the journal version, available soon!


## License

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


![Alt text](img/KCL.jpeg?raw=true "model")

![Alt text](img/DEI.jpeg?raw=true "model")

![Alt text](img/WMG.jpeg?raw=true "model")

     


