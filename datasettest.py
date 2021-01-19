
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import to_categorical, plot_model

x_list = list()
y_list = list()
x_train = list()
y_train = list()
x_test = list()
y_test = list()

# load data
for img_filename in os.listdir('training/'):
    if img_filename.endswith('.png'):
        img = load_img('training/' + img_filename, color_mode='grayscale')
        img_array = img_to_array(img)
        for i in range(6):
            x_list.append(img_array[:, i*20:(i+1)*20] / 255.0)
            y_list.append(img_filename[i])

y_list = to_categorical(y_list, num_classes=10)

# split data into training set and testing set
x_train, x_test, y_train, y_test = train_test_split(x_list, y_list, train_size=0.8)

a = np.array(x_train)

training_set_shape = a.shape
print(training_set_shape)


