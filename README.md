# Convolutional Deep Neural Network for Digit Classification

## AIM

To Develop a convolutional deep neural network for digit classification and to verify the response for scanned handwritten images.

## Problem Statement and Dataset
Digit classification and to verify the response for scanned handwritten images.

The MNIST dataset is a collection of handwritten digits. The task is to classify a given image of a handwritten digit into one of 10 classes representing integer values from 0 to 9, inclusively. The dataset has a collection of 60,000 handwrittend digits of size 28 X 28. Here we build a convolutional neural network model that is able to classify to it's appropriate numerical value.

## Neural Network Model
<img width="635" alt="jsnm" src="https://github.com/JaisonRaphael/mnist-classification/assets/94165957/0fe848b5-5bd4-4afb-bf3b-80381e39346b">

## DESIGN STEPS

### STEP 1:

Preprocess the MNIST dataset by scaling the pixel values to the range [0, 1] and converting labels to one-hot encoded format.

### STEP 2:

Build a convolutional neural network (CNN) model with specified architecture using TensorFlow Keras.

### STEP 3:

Compile the model with categorical cross-entropy loss function and the Adam optimizer.

### STEP 4:

Train the compiled model on the preprocessed training data for 5 epochs with a batch size of 64.

### STEP 5:

Evaluate the trained model's performance on the test set by plotting training/validation metrics and generating a confusion matrix and classification report. Additionally, make predictions on sample images to demonstrate model inference.

## PROGRAM

### Name: 212221230038
### Register Number: Jaison Raphael

```
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import utils
import pandas as pd
from sklearn.metrics import classification_report,confusion_matrix
from tensorflow.keras.preprocessing import image
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train.shape
X_test.shape
single_image= X_train[0]
single_image.shape
plt.imshow(single_image,cmap='gray')
X_train.min()
X_train.max()
X_train_scaled = X_train/255.0
X_test_scaled = X_test/255.0
X_train_scaled.max()

y_train[0]
y_train_onehot = utils.to_categorical(y_train,10)
y_test_onehot = utils.to_categorical(y_test,10)
type(y_train_onehot)
y_train_onehot.shape
single_image = X_train[500]
plt.imshow(single_image,cmap='gray')
y_train_onehot[500]
y_train_onehot[500]
model = keras.Sequential()
model.add(layers.Input(shape=(28,28,1)))
model.add(layers.Conv2D(filters=32,kernel_size=(3,3),activation='relu'))
model.add(layers.MaxPool2D(pool_size=(2,2)))
model.add(layers.Flatten())
model.add(layers.Dense(32,activation='relu'))
model.add(layers.Dense(10,activation='softmax'))
model.summary()
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics='accuracy')
model.fit(X_train_scaled ,y_train_onehot, epochs=5,
          batch_size=64, 
          validation_data=(X_test_scaled,y_test_onehot))
metrics = pd.DataFrame(model.history.history)
metrics.head()
metrics[['accuracy','val_accuracy']].plot()
metrics[['loss','val_loss']].plot()
x_test_predictions = np.argmax(model.predict(X_test_scaled), axis=1)
print(confusion_matrix(y_test,x_test_predictions))
print(classification_report(y_test,x_test_predictions))
img = image.load_img('images.png')
type(img)
img = image.load_img('images.png')
img_tensor = tf.convert_to_tensor(np.asarray(img))
img_28 = tf.image.resize(img_tensor,(28,28))
img_28_gray = tf.image.rgb_to_grayscale(img_28)
img_28_gray_scaled = img_28_gray.numpy()/255.0
x_single_prediction = np.argmax(
    model.predict(img_28_gray_scaled.reshape(1,28,28,1)),
     axis=1)
print(x_single_prediction)
plt.imshow(img_28_gray_scaled.reshape(28,28),cmap='gray')
img_28_gray_inverted_scaled = img_28_gray_inverted.numpy()/255.0
img_28_gray_inverted = 255.0-img_28_gray
x_single_prediction = np.argmax(
    model.predict(img_28_gray_inverted_scaled.reshape(1,28,28,1)),
     axis=1)
print(x_single_prediction)
```

## OUTPUT

### Training Loss, Validation Loss Vs Iteration Plot

![op1](https://github.com/JaisonRaphael/mnist-classification/assets/94165957/4dea8901-5d2e-48f6-83c8-b384874b1472)

<img width="400" alt="Jsn" src="https://github.com/JaisonRaphael/mnist-classification/assets/94165957/ba57562e-3be2-4bd1-a325-c41f86614328">

![op2](https://github.com/JaisonRaphael/mnist-classification/assets/94165957/1bb060df-05b9-4cbb-8435-abc5932ad8ad)

<img width="400" alt="Jsn" src="https://github.com/JaisonRaphael/mnist-classification/assets/94165957/99fd3a5e-05f0-4ab7-9b5c-6cfce0e6822a">


### Classification Report

![op3](https://github.com/JaisonRaphael/mnist-classification/assets/94165957/b1030198-bf88-4cad-a126-1720281a91ab)
<img width="400" alt="Jsn" src="https://github.com/JaisonRaphael/mnist-classification/assets/94165957/18104f5b-5d20-405a-b6ba-b10a3be29ea1">


### Confusion Matrix

![op4](https://github.com/JaisonRaphael/mnist-classification/assets/94165957/c6824a58-9efa-4a0b-8367-6a5744bb4cfb)
<img width="400" alt="Jsn" src="https://github.com/JaisonRaphael/mnist-classification/assets/94165957/6c0a8db8-a6b7-4018-9a25-45f133fc9692">


### New Sample Data Prediction
![newsampledata](https://github.com/JaisonRaphael/mnist-classification/assets/94165957/10d97ba4-87dc-434a-8a35-f52fa0b1c82b)
<img width="400" alt="Jsn" src="https://github.com/JaisonRaphael/mnist-classification/assets/94165957/24bc6d06-0fd6-4efa-979e-3caeef5c265e">

![op65](https://github.com/JaisonRaphael/mnist-classification/assets/94165957/9a0bfc92-59af-408c-bd26-a9af5e8d84fe)
<img width="400" alt="Jsn" src="https://github.com/JaisonRaphael/mnist-classification/assets/94165957/b7093d46-f4bc-4aaa-b2f8-fd4c5e771c26">


## RESULT

A convolutional deep neural network for digit classification and to verify the response for scanned handwritten images is developed succesfully
