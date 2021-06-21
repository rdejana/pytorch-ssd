# SSD-based Object Detection in PyTorch

Forked to support simple example for Homework 6.

## Prereqs
This assumes you've trained a custom object detection model using the instructions from https://github.com/dusty-nv/jetson-inference/blob/master/docs/pytorch-ssd.md.  Take note of where you've installed the jetson_inference directory; this will be referred to as $JI. 

Copy files X and Y to directoy Z.

## Native Pytorch



## TensorRT


This repo implements [SSD (Single Shot MultiBox Detector)](https://arxiv.org/abs/1512.02325) in PyTorch for object detection, using MobileNet backbones.  It also has out-of-box support for retraining on Google Open Images dataset.  

> For documentation, please refer to Object Detection portion of the **[Hello AI World](https://github.com/dusty-nv/jetson-inference/tree/dev#training)** tutorial:
> [Re-training SSD-Mobilenet](https://github.com/dusty-nv/jetson-inference/blob/dev/docs/pytorch-ssd.md)

Thanks to @qfgaohao for the upstream implementation from:  [https://github.com/qfgaohao/pytorch-ssd](https://github.com/qfgaohao/pytorch-ssd)

