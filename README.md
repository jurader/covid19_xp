# Deep learning model for classification between COVID-19 pneumonia, non-COVID-19 pneumonia, and the healthy on chest X-ray image

This repository includes source code of deep learning model for classification between COVID-19 pneumonia, non-COVID-19 pneumonia, and the healthy on chest X-ray image. The detail of the model is described in our paper (https://doi.org/10.1038/s41598-020-74539-2).

Our results show that the three-category accuracy of the model was 83.6% between COVID-19 pneumonia, non-COVID-19 pneumonia, and the healthy. Sensitivity of COVID-19 pneumonia was more than 90%. 


# Usage
The source code is written in Python (version 3.7). 
CUDA-10, tensroflow-1.13.1, and keras-2.2.4 are required for training and testing the model. 

To run the model, jupyter notebook can be used for the model. 
We recommend to use the ipynb files of this repository for running the model. 
Please upload ipynb files of this repository to Google Colaboratory or etc.


# Dataset 
Our dataset for training and testing the model is available from the following URL: https://www.dropbox.com/s/urb8kwnd6sigom5/preprocessed.tar?dl=0 

Our dataset contains 215, 533, and 500 chest X-ray images of COVID-19 pneumonia, non-COVID-19 pneumonia, and the healthy. 
These images are collected from http://arxiv.org/abs/2003.11597 and https://www.kaggle.com/c/rsna-pneumonia-detection-challenge.


# Paper 
If the source code of this repository is used, please cite the following paper.

Nishio, M., Noguchi, S., Matsuo, H. et al. Automatic classification between COVID-19 pneumonia, non-COVID-19 pneumonia, and the healthy on chest X-ray image: combination of data augmentation methods. Sci Rep 10, 17532 (2020). https://doi.org/10.1038/s41598-020-74539-2



# License
The source code of this repository is licensed under GPLv3. 
For the detail, please see License file of this repository. 


