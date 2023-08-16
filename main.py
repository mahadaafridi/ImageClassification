import tensorflow as tf
import os

import cv2
import imghdr

#this is used to prevent out of memory errors
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus: #go through every gpus inside of it and run this on every one of them
    tf.config.experimental.set_memory_growth(gpu, True)

# pshows how many gpus there are available
# gpus = tf.config.experimental.list_physical_devices('GPU')
# num_gpus = len(gpus)


data_dir = 'data'
image_exts = ['jpeg', 'jpg', 'bmp', 'png']

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
print('works')