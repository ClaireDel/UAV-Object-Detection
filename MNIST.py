import tensorflow as tf
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from tensorflow.keras.utils import to_categorical
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from numpy import argmax

# Dataset
mnist = tf.keras.datasets.mnist
(training_images, training_labels), (test_images, test_labels) = mnist.load_data()

# Normalization
training_images  = training_images / 255.0
test_images = test_images / 255.0

# Visualization
plt.imshow(test_images[0], cmap=plt.get_cmap('gray'))
plt.show()


## MODEL A ..........................................................

class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get("loss")<0.01):
      print("\nReached 99% accuracy so cancelling training!")
      self.model.stop_training = True

# ....................................................................

# Model
callbacks = myCallback()

model = tf.keras.models.Sequential([tf.keras.layers.Flatten(),
                                    tf.keras.layers.Dense(1024, activation=tf.nn.relu),
                                    tf.keras.layers.Dense(10, activation=tf.nn.softmax)]) # 10 neurons for 10 classes

model.compile(optimizer = 'Adam', loss = 'sparse_categorical_crossentropy', metrics=['accuracy'])

history = model.fit(training_images, training_labels, validation_data=(test_images, test_labels), 
                    epochs=5, 
                    callbacks=[callbacks]) 
# avoid overfitting : not too much epochs

model.evaluate(test_images, test_labels)

classifications = model.predict(test_images)

print(classifications[0]) # probability that this item 0 is each of the 10 classes
print(test_labels[0])

# ....................................................................


# Evaluation
# Overall:
train_acc = model.evaluate(training_images, training_labels, verbose=1)
test_acc = model.evaluate(test_images, test_labels, verbose=1)
# print('Train: %.3f, Test: %.3f' % (train_acc, test_acc))

# Plot training history
plt.plot(history.history['loss'], label='training')
plt.title('Training curve (loss)')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.show()

# Plot training history
plt.plot(history.history['accuracy'], label='training')
plt.title('Training curve (accuracy)')
plt.ylabel('accuracy')
plt.xlabel('Epoch')
plt.legend()
plt.show()

# ....................................................................

# Validation
acc=history.history['accuracy']
val_acc=history.history['val_accuracy']
loss=history.history['loss']
val_loss=history.history['val_loss']

epochs=range(len(acc)) # Get number of epochs

# Plot training and validation accuracy per epoch
# Tr and Val accuracy curves
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.ylim(0.7,1)
plt.title('Training and validation accuracy curves')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# Plot training and validation loss per epoch
# Tr and Val accuracy curves
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.ylabel('loss')
plt.xlabel('epoch')
plt.ylim(0,0.5)
plt.title('Training and validation loss curves')
plt.legend(['train', 'test'], loc='upper left')
plt.show()




## MODEL B .........................................................

# Dataset
mnist = tf.keras.datasets.mnist
(training_images, training_labels), (test_images, test_labels) = mnist.load_data()

# Shape
training_images = training_images.reshape((training_images.shape[0], 28, 28, 1))
test_images = test_images.reshape((test_images.shape[0], 28, 28, 1))

# One hot encode target values
training_labels = to_categorical(training_labels)
test_labels = to_categorical(test_labels)
training_images = training_images.astype('float32')
training_labels = training_labels.astype('float32')
test_images = test_images.astype('float32')
test_labels = test_labels.astype('float32')

# Sizes 
print('Train: X=%s, y=%s' % (training_images.shape, training_labels.shape))
print('Test: X=%s, y=%s' % (test_images.shape, test_labels.shape))

# Normalization
training_images  = training_images / 255.0
test_images = test_images / 255.0

# ....................................................................

# Model
model = tf.keras.models.Sequential([tf.keras.layers.Conv2D(32, (3, 3), activation=tf.nn.relu, 
                                                           kernel_initializer='he_uniform', 
                                                           input_shape=(28, 28, 1)),
                                    #tf.keras.layers.BatchNormalization(),
                                    tf.keras.layers.MaxPooling2D((2, 2)),
                                    tf.keras.layers.Conv2D(64, (3, 3), activation=tf.nn.relu, 
                                                           kernel_initializer='he_uniform'),
                                    tf.keras.layers.Conv2D(64, (3, 3), activation=tf.nn.relu, 
                                                           kernel_initializer='he_uniform'),
                                    tf.keras.layers.MaxPooling2D((2, 2)),
                                    tf.keras.layers.Flatten(),
                                    tf.keras.layers.Dense(100, activation=tf.nn.relu, 
                                                              kernel_initializer='he_uniform'),
                                    #tf.keras.layers.BatchNormalization(),
                                    tf.keras.layers.Dense(10, activation=tf.nn.softmax)]) 

model.compile(optimizer = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9),
              loss = 'categorical_crossentropy', 
              metrics=['accuracy'])

history = model.fit(training_images, training_labels,  epochs=10, batch_size=32, 
                    validation_data=(test_images, test_labels), verbose=0) 

# ....................................................................

# Precision
_, acc = model.evaluate(test_images, test_labels, verbose=0)
print('> %.3f' % (acc * 100.0))

# ....................................................................

# Evaluation
# Overall:
train_acc = model.evaluate(training_images, training_labels, verbose=1)
test_acc = model.evaluate(test_images, test_labels, verbose=1)
# print('Train: %.3f, Test: %.3f' % (train_acc, test_acc))

# Plot training history
plt.plot(history.history['loss'], label='training')
plt.title('Training curve (loss)')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.show()

# Plot training history
plt.plot(history.history['accuracy'], label='training')
plt.title('Training curve (accuracy)')
plt.ylabel('accuracy')
plt.xlabel('Epoch')
plt.legend()
plt.show()

# ....................................................................

# Validation 
acc=history.history['accuracy']
val_acc=history.history['val_accuracy']
loss=history.history['loss']
val_loss=history.history['val_loss']

epochs=range(len(acc)) # Get number of epochs

# Plot training and validation accuracy per epoch
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.ylim(0.8,1)
plt.title('Training and validation accuracy curves')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# Plot training and validation loss per epoch
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.ylabel('loss')
plt.xlabel('epoch')
plt.ylim(0,0.5)
plt.title('Training and validation loss curves')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# Prediction test 1
img = test_images[0] 
plt.imshow(img, cmap=plt.get_cmap('gray'))
plt.show()
img = img.reshape(1, 28, 28, 1)

# Predict the class
predict_value = model.predict(img)
digit = argmax(predict_value)
print(digit)


# Prediction test 2
img = load_img("2.jpg", grayscale=True, target_size=(28, 28))
plt.imshow(img, cmap=plt.get_cmap('gray'))
plt.show()

# Convert 
img = img_to_array(img)
img = img.reshape(1, 28, 28, 1)
img = img.astype('float32')
img = img / 255.0

#img = img.reshape((img.shape[0], 28, 28, 1))
#img = to_categorical(img)

predict_value = model.predict(img)
digit = argmax(predict_value)
print(digit)