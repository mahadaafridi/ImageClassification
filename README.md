# Image Classification with TensorFlow and Keras

## Overview

This project demonstrates image classification using a Convolutional Neural Network (CNN) implemented with TensorFlow and Keras. The goal is to classify images into one of two classes: "Depressed" or "Angry." The model is trained to recognize patterns and features in images to make these classifications.

## Table of Contents

- [Dataset](#Dataset)
- [Model Architecture](#Model-Architecture)
- [Training](#Training)
- [Evaluation](#Evaluation)
- [Usage](#Usage)


## Dataset
The Data will be contained in the Data folder and images put into their respective type, "Angry" or "Depressed".

## Model-Architecture

The model architecture consists of Conv2D layers, MaxPooling2D layers, and Dense layers. The specific architecture, hyperparameters, and training details are available in the Python script.

## Training
To train the data run this
python train.py --data_dir data/ --epochs 20

## Evaluation

Evaluate the code like this
python evaluate.py --model_path model.pth --test_data test_data/

# Usage
Run this code on new images to classify.

img = cv2.imread('new_image.jpg')
resize = tf.image.resize(img, (256, 256))
yhat = model.predict(np.expand_dims(resize / 255, 0))
if yhat > 0.5:
    print(f'Predicted class is Depressed')
else:
    print(f'Predicted class is Angry')




