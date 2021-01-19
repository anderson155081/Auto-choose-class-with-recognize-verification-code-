import numpy as np
from tensorflow.keras import models
from tensorflow.keras.preprocessing.image import load_img, img_to_array

np.set_printoptions(suppress=True, linewidth=150, precision=9, formatter={'float': '{: 0.9f}'.format})

# load model
model = models.load_model('cnn_model.h5')

# load img to predict
img_filename = input('Varification code img filename: ')
img = load_img(img_filename, color_mode='grayscale')
img_array = img_to_array(img)

# split the 6 digits
x_list = list()
for i in range(6):
    x_list.append(img_array[:, i*20:(i+1)*20] / 255.0)

# predict
print(model.predict(np.array(x_list)))
print(model.predict_classes(np.array(x_list)))
