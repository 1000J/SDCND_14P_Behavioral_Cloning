# -*- coding: utf-8 -*-
"""
Created on Sun Mar  4 20:00:25 2018

@author: home
"""

import csv
import cv2
import numpy as np
import sklearn

import os
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from keras.models import Sequential
from keras.layers import Convolution2D, Conv2D, ELU, Cropping2D, Flatten, Dense, Dropout, Lambda, Activation, MaxPooling2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.optimizers import Adam

DATA_PATH = './sim_data/'
LOG_SPLIT = '\\'
#DATA_PATH = './data/'
#LOG_SPLIT = '/'

    
def training_model():
    row, col, depth = 160, 320, 3
#    row, col, depth = 320, 160, 3
    model = Sequential()
    model.add(Cropping2D(cropping=((70,25),(0,0)), input_shape = (row, col, depth)))

    # normalization
    model.add(Lambda(lambda x: (x / 255) - 0.5))

    #model nvidia autonomous driving team with drop out added at FC layers
    model.add(Convolution2D(24,5,5,subsample=(2,2),activation='relu'))
    model.add(Convolution2D(36,5,5,subsample=(2,2),activation='relu'))
    model.add(Convolution2D(48,5,5,subsample=(2,2),activation='relu'))
    model.add(Convolution2D(64,3,3,activation='relu'))
    model.add(Convolution2D(64,3,3,activation='relu'))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dropout(.2))
    model.add(Activation('relu'))
    model.add(Dense(50))
    model.add(Dropout(.5))
    model.add(Activation('relu'))
    model.add(Dense(1))

#    adam = Adam(lr=0.001)  #default
    adam = Adam(lr=0.0001)  
    model.compile(optimizer = adam, loss = "mse")
#    model.summary()
    
    return model

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while True: # Loop forever so the generator never terminates
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                name = DATA_PATH + 'IMG/'+batch_sample[0].split(LOG_SPLIT)[-1]
                center_image_bgr = cv2.imread(name)
                center_image = cv2.cvtColor(center_image_bgr, cv2.COLOR_BGR2RGB)
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)


def train_generator(samples, batch_size=32):
    num_samples = len(samples)
    while True: # Loop forever so the generator never terminates
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                name = DATA_PATH + 'IMG/'+batch_sample[0].split(LOG_SPLIT)[-1]
                center_image_bgr = cv2.imread(name)
                center_image = cv2.cvtColor(center_image_bgr, cv2.COLOR_BGR2RGB)
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)
                
            aug_images, aug_angles = [],[]
            for image, angle in zip(images, angles):
                aug_images.append(image)
                aug_angles.append(angle)
                aug_images.append(cv2.flip(image,1))
                aug_angles.append(angle*-1.0)

            # trim image to only see section with road
            X_train = np.array(aug_images)
            y_train = np.array(aug_angles)
            yield sklearn.utils.shuffle(X_train, y_train)

def valid_generator(samples, batch_size=32):
    num_samples = len(samples)
    while True: # Loop forever so the generator never terminates
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                name = DATA_PATH + 'IMG/'+batch_sample[0].split(LOG_SPLIT)[-1]
                center_image_bgr = cv2.imread(name)
                center_image = cv2.cvtColor(center_image_bgr, cv2.COLOR_BGR2RGB)
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)


def data_nongenerator(lines):
    
    images = []
    measurements = []
    k = 0
    for line in lines:
        k += 1
        if k in [1, 100, 500, 1000, 3000, 7000]: print ('---> img # ',k)
        
        for i in range(3):
            source_path = line[i]
            filename = source_path.split('/')[-1]
            #        current_path = 'C:/courses/udacity/SDCND/14_project_behavioral_cloning/CarND-Behavioral-Cloning-P3/sim_data/IMG/' + filename
            current_path = DATA_PATH + 'IMG/' + filename
            image_bgr = cv2.imread(current_path)
            image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            images.append(image)
            measurement = float(line[3])
            measurements.append(measurement)
                
    augmented_images, augmented_measurements = [], []
    for image, measurement in zip(images,measurements):
        augmented_images.append(image)
        augmented_measurements.append(measurement)
        augmented_images.append(cv2.flip(image,1))
        augmented_measurements.append(measurement*-1.0)
                
                    
    X_train = np.array(augmented_images)
    y_train = np.array(augmented_measurements)

    return X_train, y_train



if __name__ == "__main__":

    cwd = os.getcwd()
    print(cwd)

    lines = []

    print('-------> Reading the driving_log.csv')
    #with open('C:/courses/udacity/SDCND/14_project_behavioral_cloning/CarND-Behavioral-Cloning-P3/sim_data/driving_log.csv') as csvfile:
    with open(DATA_PATH + 'driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader: 
            if line[0] != 'center':
                lines.append(line)

    print('length of lines ', len(lines))
    print('-------> Getting data')

    # non-generator method
    # X_train, y_train = data_nongenerator(lines)
    # print('-------> Start training')    
    # model = training_model()
    # model.fit(X_train, y_train, validation_split=0.2,shuffle=True, nb_epoch=5)

    # compile and train the model using the generator function
    # randomly split the data into training and validation set
    
    train_samples, validation_samples = train_test_split(lines, test_size=0.2)
    train_generator = train_generator(train_samples, batch_size=32)
    validation_generator = valid_generator(validation_samples, batch_size=32)

    model = training_model()
    model.fit_generator(train_generator, samples_per_epoch=len(train_samples)*2, 
                        validation_data=validation_generator,
                        nb_val_samples=len(validation_samples), nb_epoch=10)

    model.save('model.h5')
