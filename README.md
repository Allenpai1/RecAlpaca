# CASA0018 Project - Image Open Domain QA system via GPT3 API, Cameras and Raspberry Pi 4B

### CASA0018 Project - Image Open Domain QA system via GPT3 API, Cameras and Raspberry Pi 4B

### The detailed project description can be find in [my report](./report/report.md).

<br />

> The system can be trained on PyTorch and TensorFlow, and for both platforms, the source code of Faster-RCNN are included.  Some issues remained unsolved during the experiment on my M1 macs with [TensorFlow object detection environments](https://github.com/tensorflow/io/issues/1625), so PyTorch was used for the primary results production.

## System Overview

For this project, I have build a multilingual language (Chinese and English) image-to-text open domain QA system with the [ChatGPT-API](https://openai.com/blog/introducing-chatgpt-and-whisper-apis) and [Paddle OCR](https://github.com/PaddlePaddle/PaddleOCR). The system consists of 3 steps:
 1. Language and text detection: a detection model Faster-RCNN is used to recognize the text from the image and language of that text(Chinese or English) and crop the text from the original image with the language classification results passed into downstream tasks;
 2. Convert zoomed image to text: the previous step results are passed into paddleOCR, and the corresponding language text recognition model is loaded to extract the text from zoomed text image;
 3. QA system: ChatGPT API is used to pass the detected text and return a detailed answer, saved as PDF format and JPG images.

![plot](./Images/system2.png)

I tried the handwritten text recognition models on the open-source library KerasOCR; it turns out that the KeraOCR cannot recognize handwritten text, and only English language recognition models are supported. This is because KerasOCR is primarily designed to recognize printed text. 

Therefore to allow muti-language handwritten text detections, I have trained two-stage object detection Faster-RCNN models to detect Chinese or English handwriting questions text from images. The dataset constructed by myself contains my handwritten text from ChatGPT history questions. The detected text image is cropped, and the language classification results pass into PaddleOCR to load the crossposting text extraction model. After that, the detected texts are used as input to pass into the ChatGPT model through an API connection to return an answer. The program will output intermediate step results and save the final Question&Answer to a pdf file. 

## Experiments Environments
 - OS: Intel(R) Core(TM) i5-8259U@230 GHz Macbook pro 2020
 - Raspberry Pi 4B 4GB OS(64-bit)
 - Platform: Python 3.10.8, pytorch 1.13.1, tensorflow 2.11.1, paddleocr 2.6.1.0, openai 0.27.2, OpenCV 4.5.5
 - GPU:A100-SXM4-80GB hired on [AutoDL](https://www.autodl.com/home)
 - IPhone with [IP Camera](https://github.com/shenyaocn/IP-Camera-Bridge) App

## Camera connection and data creation
 - For ground truth object bounding box creation, I have used [labelImg](https://github.com/heartexlabs/labelImg).
 - To connect the iPhone camera to the laptop [IP camera](https://github.com/shenyaocn/IP-Camera-Bridge) was used.
 
## Install requirements
 - ```pip install -r requirements.txt```
 - The pretrained restnet50 backbone model on [VOC dataset](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/) can be donwload on [google drive](https://drive.google.com/drive/folders/1bBdFgyOmAyaZJotF_79pKl8VIXhYggee?usp=sharing) and include in Pytorch\ train\ F-RCNN/model_data for training.
 - The full dataset can be download on [Own dataset](https://drive.google.com/drive/folders/1d7Cq-iJxVMWsWlyYrQ-pGN5UGvLxXnRg?usp=sharing).

## My Faster-RCNN Detection training Results
Many experiments are done on the model experiments; please see [my report](./report/report.md) for more details. I have listed the most interesting ones, where the model trained from scratch is slightly less performed and take long training times than with the backbone resnet50 pre-trained model loaded. The best model achieves a 90.02% AmAP. (Also, try not to train a deep model on the CPU).
![plot](./report/compare.png)

## Usage and run
The code under the folder System code is already defined and ready to use, where the main.py is the system connected with the camera and QA_results.pdf is the returning pdf results; it will get updated once you run.

To run the program you can download my pretrained model on [google drive](https://drive.google.com/drive/folders/1bBdFgyOmAyaZJotF_79pKl8VIXhYggee?usp=sharing) and put it in System\ code/logs folder and in the main.py program you need to generate a personal [ChatGPT-API key](https://openai.com/blog/introducing-chatgpt-and-whisper-apis) :
```
 cd System\ code/
 python main.py
```
When the camera is launched, press the ```space``` keyword to take the photo and ```q``` to quit the system.
The system will return the intermediate results, the results will be similar as following:
 1. For Faster-RCNN text language detection:
 
 ![plot](./Images/combine.png)

 2. PaddleOCR text extraction results:
 ![plot](./Images/paddleChinese.png)
 ![plot](./Images/paddleEnglish.png)

 3. QA results in PDF:
 ![plot](./Images/ChineseDoc.png)
 ![plot](./Images/EnglishDoc.png)

 ## Train you own model
Ensure the dataset is in the correct VOC format and put it into the folder PyTorch train F-RCNN and run ```python train.py``` and you need to change your own model_data/classes.txt. It would be beneficial to understand how the network architecture is written by looking at the source code I have included. The source codes are collected from [faster-rcnn-pytorch](https://github.com/bubbliiiing/faster-rcnn-pytorch) and [tensorflow-faster-rcnn](https://github.com/endernewton/tf-faster-rcnn).
![plot](./Images/FRCNN.png)
