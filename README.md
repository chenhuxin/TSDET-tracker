# TSDET-tracker
 Deep Ensemble Object Tracking Based on Temporal and Spatial Networks
## This is the implementation of our TSDET paper. The paper page can be found here: (https://www.kaggle.com/c/aerial-cactus-identification)

There are three folders in this repository where the matconvnet toolbox and Skiing sequences are contained. Our main development is kept in the folder CREST.

Before running our code, check if you have a state-of-the-art GPU. I develop this code using Titan Black. Make sure yours are better than mine :-).

Please download the VGG-16 model and put it under 'CREST/exp/model/'. You can download VGG-16 model via http://www.vlfeat.org/matconvnet/pretrained/.

Meanwhile, please configure matconvnet on your side. (You need to compile matconvnet using the provided because of the modifications.)

Try 'CREST/demo.m' to see the tracker performance on the Skiing sequences.

If you find the code useful, please cite:
