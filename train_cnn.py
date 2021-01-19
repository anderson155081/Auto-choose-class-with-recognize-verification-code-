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

# model
model = Sequential()
model.add(Conv2D(8, kernel_size=(3, 3), activation='relu', input_shape=(32, 20, 1)))
model.add(Conv2D(16, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dense(10, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])

plot_model(model, to_file='model.png', show_shapes=True, dpi=300)

epochs = 100

# start training
history = model.fit(np.array(x_train), np.array(y_train), epochs=epochs, validation_data=(np.array(x_test), np.array(y_test)))

# evaluate model
loss, accuracy = model.evaluate(np.array(x_test), np.array(y_test))
print('Test loss:', loss)
print('Test accuracy:', accuracy)

# save the model
model.save('cnn_model.h5')

# plot the training results using matplotlib
loss = history.history['loss']
acc = history.history['accuracy']
val_loss = history.history['val_loss']
val_acc = history.history['val_accuracy']

fig = plt.figure(figsize=(12, 4))

fig.add_subplot(1, 2, 1)
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend()

fig.add_subplot(1, 2, 2)
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend()

plt.show()
