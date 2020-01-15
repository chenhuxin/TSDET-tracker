# TSDET-tracker: Deep Ensemble Object Tracking Based on Temporal and Spatial Networks 

## Huxin Chen
This is the implementation of our TSDET paper. The paper page can be found here: https://ieeexplore.ieee.org/document/8950038

## Requirements
There is one folder in this repository. Our main development is kept in the TSDET folder. Matconvnet folder and test sequence are placed in dataset folder and external folder, respectively.

1. MatConvNet-1.0-beta24 or latest
2. Matlab2017b
3. VS2015
4. Cuda8.0
5. cudnnV5
6. git clone https://github.com/chenhuxin/TSDET-tracker.git

## How to run the Code
Before running our code, check if you have a state-of-the-art GPU. I develop this code using NVIDIA GTX 1060. Make sure yours are better than mine.

1. Please download the VGG-16 model and put it under 'TSDET/exp/model/'. You can download VGG-16 model via http://www.vlfeat.org/matconvnet/pretrained/.
2. Compile the MatConvNet according to the [website](http://www.vlfeat.org/matconvnet/install/).
3. Try `TSDET/demo.m` to see the tracker performance on the test sequence.

## Results
### Results on OTB-2015
![Image text](https://github.com/chenhuxin/TSDET-tracker/results/OTB-2015.png)  

## Contact
Huxin Chen <br>
Email: huxin_chen@163.com
