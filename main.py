import tensorflow as tf
import os

import cv2
import imghdr

import numpy as np
import matplotlib.pyplot as plt

#this is used to prevent out of memory errors
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus: #go through every gpus inside of it and run this on every one of them
    tf.config.experimental.set_memory_growth(gpu, True)

# pshows how many gpus there are available
# gpus = tf.config.experimental.list_physical_devices('GPU')
# num_gpus = len(gpus)


data_dir = 'data'
image_exts = ['jpeg', 'jpg', 'bmp', 'png']

#this part handles checking image types and errors that might occur with processing images
for image_class in os.listdir(data_dir): #this will loop through each folder
    #loop through each image in the subdirectery
    for image in os.listdir(os.path.join(data_dir, image_class)):
        image_path = os.path.join(data_dir, image_class, image)
        try:
            #opens up the image as a numpy array
            img = cv2.imread(image_path)
            tip = imghdr.what(image_path)
            if tip not in image_exts:
                print('Image not in ext list {}'.format(image_path))
                os.remove(image_path)
        except Exception as e:
            print('Issue with image {}'.format(image_path))

#this will preprocess image data
#'data' is the path to the file
#data a generator that will store the dataset
data = tf.keras.utils.image_dataset_from_directory('data')


#this allows us to access our generator
data_iterator = data.as_numpy_iterator()

batch = data_iterator.next()


fig, ax = plt.subplots(ncols=4, figsize=(20,20)) #creates a set of subplots, where 4 will be in a single row
for idx, img in enumerate(batch[0][:4]): #iterates through each of the four images
    ax[idx].imshow(img)
    ax[idx].title.set_text(batch[1][idx]) #sets the title to the current image


# preprocess the data

#allows us to transform the data to improve the performance of the neural  network
data = data.map(lambda x,y:(x/255, y)) #x is the images, y is the labels


train_size = int(len(data)* .7)
val_size = int(len(data)*.2)
test_size = int(len(data)*.1) * 1