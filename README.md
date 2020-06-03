# Deep learning model for classification between COVID-19 pneumonia, non-COVID-19 pneumonia, and the healthy on chest X-ray image

This repository includes source code of deep learning model for classification between COVID-19 pneumonia, non-COVID-19 pneumonia, and the healthy on chest X-ray image. The detail of the model is described in our preprint paper (https://arxiv.org/abs/2006.00730).

Our results show that the three-category accuracy of the model was 83.6% between COVID-19 pneumonia, non-COVID-19 pneumonia, and the healthy. Sensitivity of COVID-19 pneumonia was more than 90%. 

# Usage
CUDA10, tensroflow-1.13.1, and keras-2.2.4 are required for training and testing the model. 

To run the model, jupyter notebook can be used for the model. 
We recommend to use the ipynb files of this repository for running the model. 
Please upload ipynb files of this repository to Google Colaboratory or etc.


# Dataset 
Our dataset for training and testing the model is available from the following URL: https://www.dropbox.com/s/urb8kwnd6sigom5/preprocessed.tar?dl=0 

Our dataset contains 215, 533, and 500 chest X-ray images of COVID-19 pneumonia, non-COVID-19  pneumonia,  and  the  healthy. 
These images are collected from http://arxiv.org/abs/2003.11597 and https://www.kaggle.com/c/rsna-pneumonia-detection-challenge.


# Paper 
If source code of repository is used, please cite the following paper.

Automatic classification between COVID-19 pneumonia, non-COVID-19 pneumonia, and the healthy on chest X-ray image: combination of data augmentation methods in a small dataset.
Mizuho Nishio, Shunjiro Noguchi, Hidetoshi Matsuo, Takamichi Murakami.

https://arxiv.org/abs/2006.00730


# License
Source code of this repository was licensed under GPLv3. 
For the detail, please see License file of this repository. 


