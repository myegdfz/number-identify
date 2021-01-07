import os
from keras.datasets import mnist
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
path = r'E:\\Image Identification\\DataSet\\OwnMade\\ceshi\\'
files = os.listdir(path)
list_img_3d = []
list_img_label = []
fals = 1
for file in files:
    if "DS_Store" in file:
        continue
    img = Image.open(path + file).convert("L")
    list_img_label.append(np.array(int(file[3])))
    list_img_3d.append(np.array(img))

arr_img_3d = np.array(list_img_3d)
arr_img_lable = np.array(list_img_label)
np.savez(os.path.join("test_mnist.npz"), features=arr_img_3d, labels=arr_img_lable)

(_x, _y), (x_test, y_test) = mnist.load_data('./mnist.npz')
print(_x.shape)

data_delta = np.load("test_mnist.npz")
print(data_delta['labels'][200])
print(data_delta['labels'].shape)
# print(data_delta['features'][0].shape)
plt.imshow(data_delta['features'][200])
plt.show()