# Import Library
import numpy as np 
import matplotlib.pyplot as plt
from glob import glob
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.applications.vgg16 import VGG16
import os

import warnings
warnings.filterwarnings("ignore")

train_path = "./transfer-learning-dataset/indoorCVPR_09/Images"
test_path = "./transfer-learning-dataset/indoorCVPR_09_Train/Images"

# The number of classes of dataset
numberOfClass = len(glob(train_path + "/*"))
print("Number Of Class: ", numberOfClass)

# Visualize
#img = load_img(train_path + "/bedroom/000106949d.jpg")
#plt.imshow(img)
#plt.axis("off")
#plt.show()

# The images size in dataset.
#image_shape = img_to_array(img)
#print(image_shape.shape)


# Prepare the datasef for vgg16
train_data = ImageDataGenerator().flow_from_directory(train_path, target_size = (224,224))
test_data = ImageDataGenerator().flow_from_directory(test_path, target_size = (224,224))

for i in os.listdir(train_path):
    for j in os.listdir(train_path+ "/" + i):
        img = load_img(train_path+ "/" + i + "/" + j)
        plt.imshow(img)
        plt.title(i)
        plt.axis("off")
        plt.show()
        break

vgg16 = VGG16()
vgg16.summary()
vgg16_layer_list = vgg16.layers
for i in vgg16_layer_list:
    print(i)

# add the layers of vgg16 in my created model.
vgg16Model = Sequential()
for i in range(len(vgg16_layer_list)-1):
    vgg16Model.add(vgg16_layer_list[i])
    
# the final version of the model
vgg16Model.summary()

# Close the layers of vgg16
for layers in vgg16Model.layers:
    layers.trainable = False

# Last layer
vgg16Model.add(Dense(numberOfClass, activation = "softmax"))

# After I added last layer in created model.
vgg16Model.summary()

# I create compile part.
vgg16Model.compile(loss = "categorical_crossentropy",
             optimizer = "rmsprop",
             metrics = ["accuracy"])

# Traning with model
batch_size = 32

hist_vgg16 = vgg16Model.fit(train_data, 
                            steps_per_epoch=1600 // batch_size, 
                            epochs=10, 
                            validation_data=test_data, 
                            validation_steps=800 // batch_size)

# Save the weights of model
vgg16Model.save_weights("deneme.weights.h5")

# Loss and Validation Loss
plt.plot(hist_vgg16.history["loss"], label = "training loss")
plt.plot(hist_vgg16.history["val_loss"], label = "validation loss")
plt.legend()
plt.show()

# Accuracy and Validation Accuracy
plt.plot(hist_vgg16.history["accuracy"], label = "accuracy")
plt.plot(hist_vgg16.history["val_accuracy"], label = "validation accuracy")
plt.legend()
plt.show()

import json, codecs
with open("deneme.json","w") as f:
    json.dump(hist_vgg16.history, f)

with codecs.open("./deneme.json","r", encoding = "utf-8") as f:
    load_result = json.loads(f.read())

# Loss And Validation Loss
plt.plot(load_result["loss"], label = "training loss")
plt.plot(load_result["val_loss"], label = "validation loss")
plt.legend()
plt.show()



