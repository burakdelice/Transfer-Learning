import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

train_path = "./transfer-learning-dataset/indoorCVPR_09/Images"
test_path = "./transfer-learning-dataset/indoorCVPR_09_Train/Images"

numberOfClass = len(glob(train_path + "/*"))

train_data_gen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

test_data_gen = ImageDataGenerator(rescale=1./255)

train_data = train_data_gen.flow_from_directory(train_path, target_size=(224,224), batch_size=64, class_mode='categorical')
test_data = test_data_gen.flow_from_directory(test_path, target_size=(224,224), batch_size=64, class_mode='categorical')

def create_model(dropout_rate=0.0):
    model = Sequential()
    model.add(Input(shape=(224, 224, 3)))
    
    # Convolutional layers
    model.add(Conv2D(64, (3,3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    if dropout_rate > 0:
        model.add(Dropout(dropout_rate))
    
    model.add(Conv2D(128, (3,3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    if dropout_rate > 0:
        model.add(Dropout(dropout_rate))
    
    model.add(Conv2D(256, (3,3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    if dropout_rate > 0:
        model.add(Dropout(dropout_rate))
    
    model.add(Conv2D(512, (3,3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    if dropout_rate > 0:
        model.add(Dropout(dropout_rate))
    
    # Flatten layer
    model.add(Flatten())
    
    # Fully connected layers
    model.add(Dense(2048, activation='relu'))
    if dropout_rate > 0:
        model.add(Dropout(dropout_rate))
    
    model.add(Dense(1024, activation='relu'))
    if dropout_rate > 0:
        model.add(Dropout(dropout_rate))
    
    # Output layer
    model.add(Dense(numberOfClass, activation='softmax'))
    
    return model

def compile_model(model, learning_rate):
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

learning_rates = [0.001, 0.0001]
batch_sizes = [64, 128]

best_accuracy = 0
best_model = None

for lr in learning_rates:
    for bs in batch_sizes:
        model = create_model()
        model = compile_model(model, lr)
        
        history = model.fit(train_data, 
                            steps_per_epoch=len(train_data) // bs, 
                            epochs=10, 
                            validation_data=test_data, 
                            validation_steps=len(test_data) // bs)
        
        model_save_path = f'part1_lr_{lr}_bs_{bs}.h5'
        model.save(model_save_path)
        print(f'Model kaydedildi: {model_save_path}')
        
        if history.history['val_accuracy'][-1] > best_accuracy:
            best_accuracy = history.history['val_accuracy'][-1]
            best_model = model
        
        # Yalnızca en son epoch'taki doğruluk değerini gösteren grafik
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('Accuracy/Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.show()

best_model.save('best_model_part1.h5')
print('En iyi model kaydedildi.')

Y_pred = best_model.predict(test_data)
y_pred = np.argmax(Y_pred, axis=1)
y_true = test_data.classes

cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=test_data.class_indices.keys())
disp.plot(cmap=plt.cm.Blues)
plt.show()
