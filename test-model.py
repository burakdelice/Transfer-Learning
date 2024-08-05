import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import img_to_array, load_img, ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.applications.vgg16 import VGG16
import os

class ImageClassifier:
    def __init__(self, model_weights_path, train_data_path):
        self.model_weights_path = model_weights_path
        self.train_data_path = train_data_path
        self.numberOfClass = len(os.listdir(train_data_path))
        self.model = self._load_model()
        
    def _load_model(self):
        vgg16 = VGG16()
        model = Sequential()
        for i in range(len(vgg16.layers) - 1):
            model.add(vgg16.layers[i])
        
        for layer in model.layers:
            layer.trainable = False
        
        model.add(Dense(self.numberOfClass, activation="softmax"))
        model.load_weights(self.model_weights_path)
        
        return model
    
    def classify_image(self, image_path):
        img = load_img(image_path, target_size=(224, 224))
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        
        prediction = self.model.predict(img_array)
        class_index = np.argmax(prediction, axis=1)
        
        class_labels = {v: k for k, v in ImageDataGenerator().flow_from_directory(self.train_data_path, target_size=(224, 224)).class_indices.items()}
        predicted_class = class_labels[class_index[0]]
        
        plt.imshow(img)
        plt.title(f'Predicted Class: {predicted_class}')
        plt.axis("off")
        plt.show()


model_weights_path = "deneme.weights.h5"  # Model ağırlıklarının kaydedildiği dosya
train_data_path = "./transfer-learning-dataset/indoorCVPR_09/Images"  # Eğitim veri yolunu girin

classifier = ImageClassifier(model_weights_path, train_data_path)

test_folder = './test-images/'

images = os.listdir(test_folder)

for image_name in images:
   
    image_path = os.path.join(test_folder, image_name)
   
    classifier.classify_image(image_path)





