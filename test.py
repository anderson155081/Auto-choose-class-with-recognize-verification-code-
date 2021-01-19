from selenium import webdriver
import numpy as np
from tensorflow.keras import models
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from selenium.webdriver import ActionChains 
from selenium.webdriver.common.keys import Keys
import time
import os
import shutil




username=("407850105")
password=("Aall9819")


driver = webdriver.Chrome(executable_path='/usr/local/bin/chromedriver')

driver.get('https://www.ais.tku.edu.tw/EleCos/login.aspx')


# with open('filename.png', 'wb') as file:
#     file.write(driver.find_element_by_xpath('//*[@id="imgCONFM"]').screenshot_as_png)
img = driver.find_element_by_xpath('//*[@id="imgCONFM"]')
actionChains = ActionChains(driver)
actionChains.move_to_element(img).context_click().perform()
time.sleep(4)
shutil.move("/Users/chi-an/Downloads/confirm.png", "/Users/chi-an/Desktop/code/webautomation/class/confirm.png")

email = driver.find_element_by_xpath('//*[@id="txtStuNo"]')
email.send_keys(username)
secret = driver.find_element_by_xpath('//*[@id="txtPSWD"]')
secret.send_keys(password)


np.set_printoptions(suppress=True, linewidth=150, precision=9, formatter={'float': '{: 0.9f}'.format})

# load model
model = models.load_model('cnn_model.h5')



# load img to predict
img_filename = ("confirm.png")
img = load_img(img_filename, color_mode='grayscale')
img_array = img_to_array(img)

# split the 6 digits
x_list = list()
for i in range(6):
    x_list.append(img_array[:, i*20:(i+1)*20] / 255.0)

# predict
print(model.predict(np.array(x_list)))
print(model.predict_classes(np.array(x_list)))

result = (model.predict_classes(np.array(x_list))).tolist()
print(result)
val = driver.find_element_by_xpath('//*[@id="txtCONFM"]')
for i in range(len(result)):
	val.send_keys(result[i])

signin = driver.find_element_by_xpath('//*[@id="btnLogin"]')
signin.click()






