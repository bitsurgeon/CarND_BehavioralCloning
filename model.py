import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout, Cropping2D
from keras.layers.convolutional import Conv2D

# input data feeding coroutine used by Keras fit_generator() for network training
def generator(samples, batch_size=32):
    num_samples = len(samples)
    correction = 0.05
    while 1:  # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                name_center = './data/IMG/' + batch_sample[0].split('/')[-1]
                name_left   = './data/IMG/' + batch_sample[1].split('/')[-1]
                name_right  = './data/IMG/' + batch_sample[2].split('/')[-1]

                # center camera
                center_image = cv2.cvtColor(cv2.imread(name_center), cv2.COLOR_BGR2YUV)
                center_angle = float(batch_sample[3])
                # left camera
                left_image = cv2.cvtColor(cv2.imread(name_left), cv2.COLOR_BGR2YUV)
                left_angle = center_angle + correction
                # right camera
                right_image = cv2.cvtColor(cv2.imread(name_right), cv2.COLOR_BGR2YUV)
                right_angle = center_angle - correction

                # left-right flipped images
                center_image_lr = cv2.flip(center_image, 1)
                center_angle_lr = -1.0 * center_angle
                left_image_lr = cv2.flip(left_image, 1)
                left_angle_lr = -1.0 * left_angle
                right_image_lr = cv2.flip(right_image, 1)
                right_angle_lr = -1.0 * right_angle

                # add to train set
                images.append(center_image)
                angles.append(center_angle)
                images.append(left_image)
                angles.append(left_angle)
                images.append(right_image)
                angles.append(right_angle)
                images.append(center_image_lr)
                angles.append(center_angle_lr)
                images.append(left_image_lr)
                angles.append(left_angle_lr)
                images.append(right_image_lr)
                angles.append(right_angle_lr)

            X_train = np.array(images)
            y_train = np.array(angles)
            yield shuffle(X_train, y_train)

# load csv log
samples = []
with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

# split datasets for training and validation
train_samples, validation_samples = train_test_split(samples[1:], test_size=0.2)

# Set our batch size
batch_size = 32

# set training epochs
ep = 4

# camera image sizes
row, col, ch = 160, 320, 3

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)

# create network
model = Sequential()
model.add(Cropping2D(cropping=((70,25), (0,0)), input_shape=(row, col, ch)))
model.add(Lambda(lambda x: (x / 255.0) - 0.5))
model.add(Conv2D(24, kernel_size=(5, 5), strides=(2, 2), padding='valid', activation='relu'))
model.add(Conv2D(36, kernel_size=(5, 5), strides=(2, 2), padding='valid', activation='relu'))
model.add(Conv2D(48, kernel_size=(5, 5), strides=(2, 2), padding='valid', activation='relu'))
model.add(Conv2D(64, kernel_size=(3, 3), padding='valid', activation='relu'))
model.add(Conv2D(64, kernel_size=(3, 3), padding='valid', activation='relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dropout(0.5))
model.add(Dense(50))
model.add(Dropout(0.5))
model.add(Dense(10))
model.add(Dense(1))

# display network topology
model.summary()

# training network
model.compile(loss='mse', optimizer='adam')
history_object = model.fit_generator(train_generator,
                    steps_per_epoch=np.ceil(len(train_samples)/batch_size),
                    validation_data=validation_generator,
                    validation_steps=np.ceil(len(validation_samples)/batch_size),
                    epochs=ep, verbose=1)

# plot the training and validation loss for each epoch
fig, ax = plt.subplots()
x_scale = np.arange(1, ep + 1, 1)
ax.plot(x_scale, history_object.history['loss'])
ax.plot(x_scale, history_object.history['val_loss'])
ax.set_xticks(x_scale)
ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%0.3f'))
ax.set_title('model MSE loss')
ax.set_ylabel('MSE loss')
ax.set_xlabel('epoch')
ax.legend(['training set', 'validation set'], loc='upper right')
fig.savefig('examples/training_history.png')

# save model
model.save('model.h5')
