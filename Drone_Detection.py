import os
import cv2
import imutils
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.preprocessing.image import load_img
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Input,Flatten,Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.preprocessing.image import img_to_array

# IMPORTATION
base_path="/Users/clair/Desktop/"
images=os.path.sep.join([base_path,'image'])

# Filename
R1 = []
folder = "/Users/clair/Desktop/image/"
for filename in os.listdir(folder):
    R1.append(filename)

# Labels
labels = []
folder2 = 'C:\\Users\\clair\\Desktop\\label\\'

# Store Filenames 
L = []
for filename2 in os.listdir(folder2):
    L.append(str(filename2))
    
# Store raw groundtruth
for i in range(len(L)):
    folder = os.path.join(folder2, L[i])
    with open(folder) as f:
        mylist = [line.rstrip('\n') for line in f]
    a = mylist[0].split(' ')
    a.remove('0')
    labels.append(a)
    
# Format: [x y width height]
labels[139]
labels[139].remove('')

# Clean Labels 
Labels = []
for k in range(len(labels)):
    a = [float(i) for i in labels[k]]
    Labels.append(a)
Labels = np.array(Labels)
Labels = Labels.astype(int)
Labels = Labels.astype(str)

# Store filenames ang cleaned groundtruth in Z
R2 = []
R3 = []
R4 = []
R5 = []
for i in range(len(Labels)):
    R2.append(Labels[i][0])
    R3.append(Labels[i][1])
    R4.append(Labels[i][2])
    R5.append(Labels[i][3])
Z = []
for i in range(len(R1)):
    Z.append(R1[i]+','+str(R2[i])+','+str(R3[i])+','+str(R4[i])+','+str(R5[i]))

#.......................................................

# Pre-processing 
data=[]
targets=[]
filenames=[]
for row in Z:
  row=row.split(",")
  (filename,X,Y,W,H)=row
  
  imagepaths=os.path.sep.join([images,filename])
  image=cv2.imread(imagepaths)
  (h,w)=image.shape[:2]
  
  X2 = (int(X)/w)*224
  Y2 = (int(Y)/h)*224
  W2 = (int(W)/w)*224
  H2 = (int(H)/h)*224
  
  # 
  
  image=load_img(imagepaths,target_size=(224,224))
  image=img_to_array(image)

  targets.append([X2,Y2,W2,H2])
  filenames.append(filename)
  data.append(image)

#.......................................................

# Plotting groundtruth (1 image)

# Real image
# resized = image 
# cv2.rectangle(resized,(int(X),int(Y)),(int(X)+int(W),int(Y)+int(H)),(0,255,0),3)
# plt.imshow(resized)

# Image 224X224
# resized = cv2.resize(image,(224, 224))
# cv2.rectangle(resized,(int(X2),int(Y2)),(int(X2)+int(W2),int(Y2)+int(H2)),(0,255,0),3)
# plt.imshow(resized)


#.......................................................

# Normalization
data=np.array(data,dtype='float32') / 255.0
targets=np.array(targets,dtype='float32') / 224

# Split
split=train_test_split(data,targets,filenames,test_size=0.20,random_state=42)
(train_images,test_images) = split[:2]
(train_targets,test_targets) = split[2:4]
(train_filenames,test_filenames) = split[4:]


# Model
vgg=tf.keras.applications.VGG16(weights='imagenet',include_top=False,input_tensor=Input(shape=(224,224,3)))
vgg.summary()

# Transfer Learning
vgg.trainable = False
flatten = vgg.output
flatten = Flatten()(flatten)
bboxhead = Dense(256,activation="relu")(flatten) 
bboxhead = Dense(128,activation="relu")(bboxhead) 
bboxhead = Dense(64,activation="relu")(bboxhead)
bboxhead = Dense(32,activation="relu")(bboxhead)
bboxhead = Dense(4,activation="relu")(bboxhead)

model = Model(inputs = vgg.input,outputs = bboxhead)
model.summary()
opt = Adam(1e-4) 
model.compile(loss='mse',optimizer=opt,metrics=['accuracy'])
history = model.fit(train_images,train_targets,validation_data=(test_images,test_targets),
                    batch_size=32,epochs=10 ,verbose=1)

#.......................................................
# model=load_model('/Users/clair/Desktop/detect_drones_test2.h5')


# # Visualization of the results (1 image)
# x = '09-february-2017-ufa-russia-260nw-1062173864.jpg'
# imagepath='/Users/clair/Desktop/image/'+ x
# image = load_img(imagepath,target_size=(224,224))
# image = img_to_array(image) / 255.0
# image = np.expand_dims(image,axis=0)

# preds=model.predict(image)[0]
# print(model.predict(image))
# (X2,Y2,W2,H2)=preds*224
# print(X2, Y2, W2, H2)


# image=cv2.imread(imagepath)
# image=imutils.resize(image,width=600)
# (h,w)=image.shape[:2]
# X=(X2/224)*w
# Y=(Y2/224)*h
# W=(W2/224)*w
# H=(H2/224)*h
# print(X, Y, W, H)


# resized = cv2.resize(image,(224, 224))
# # cv2.rectangle(resized,(startX, startY), (endX, endY),(0,255,0),3)
# cv2.rectangle(resized,(int(X),int(Y)),(int(X)+int(W),int(Y)+int(H)),(0,255,0),3)
# plt.imshow(resized)

# cv2.rectangle(image,(int(X),int(Y)),(int(X)+int(W),int(Y)+int(H)),(0,255,0),3)
# plt.imshow(image)
# cv2.waitKey(0)

#.......................................................

model.save('detect_drones_test2.h5')

# # Plot training history
# plt.plot(history.history['loss'], label='training')
# plt.title('Training curve')
# plt.ylabel('Loss')
# plt.xlabel('Epoch')
# plt.legend()
# plt.show()

# # Tr and Val loss curves
# plt.plot(epochs, loss, 'r')
# plt.plot(epochs, val_loss, 'b')
# plt.ylim((0,0.5))
# plt.title('Training and validation loss curves')
# plt.legend(['train', 'test'], loc='upper left')
# plt.show()


# # Tr and Val accuracy curves
# plt.plot(history.history['accuracy'])
# plt.plot(history.history['val_accuracy'])
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.ylim(0,1)
# plt.title('Training and validation accuracy curves')
# plt.legend(['train', 'test'], loc='upper left')
# plt.show()

#.......................................................


def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = int(max(boxA[0], boxB[0]))
    yA = int(max(boxA[1], boxB[1]))
    xB = int(min(boxA[2], boxB[2]))
    yB = int(min(boxA[3], boxB[3]))
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (int(boxA[2]) - int(boxA[0]) + 1) * (int(boxA[3]) - int(boxA[1]) + 1)
    boxBArea = (int(boxB[2]) - int(boxB[0]) + 1) * (int(boxB[3]) - int(boxB[1]) + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return iou


# Print both bounding boxes for the image selected 
def bothboxes(indicedansZ):
  row = Z[indicedansZ].split(",")
  (filename,X,Y,W,H)=row
  
  # Pathway
  imagepaths=os.path.sep.join([images,filename])

  # Compress for the prediction
  image = load_img(imagepaths,target_size=(224,224))
  image = img_to_array(image)/255.0
  image = np.expand_dims(image,axis=0)
  
  # Prediction
  preds = model.predict(image)[0]
  (X2,Y2,W2,H2)=preds*224

  # Real image with prediction X1...
  image = cv2.imread(imagepaths)
  # image=imutils.resize(image,width=600)
  (h,w) = image.shape[:2]
  X1 = (X2/224)*w
  Y1 = (Y2/224)*h
  W1 = (W2/224)*w
  H1 = (H2/224)*h
  
  cv2.rectangle(image,(int(X),int(Y)),(int(X)+int(W),int(Y)+int(H)),(0,255,0),3) # groundtruth
  cv2.rectangle(image,(int(X1),int(Y1)),(int(X1)+int(W1),int(Y1)+int(H1)),(255,0,0),3)
  
  iou = bb_intersection_over_union(np.array([int(X),int(Y),int(W),int(H)]), np.array([int(X1),int(Y1),int(W1),int(H1)]))

  font = cv2.FONT_HERSHEY_SIMPLEX
  org = (20, 30)
  fontScale = 1
  color = (0, 0, 0)
  thickness = 2
  image = cv2.putText(image, "IoU: {:.4f}".format(iou), org, font, fontScale, color, thickness, cv2.LINE_AA)
  
  plt.imshow(image)
  cv2.waitKey(0)
  
  
def groundtruth(indicedansZ):
  row = Z[indicedansZ].split(",")
  (filename,X,Y,W,H)=row 
  imagepaths=os.path.sep.join([images,filename])
  image=cv2.imread(imagepaths)
  # Print groundtruth bounding boxes
  cv2.rectangle(image,(int(X),int(Y)),(int(X)+int(W),int(Y)+int(H)),(0,255,0),3) # grountruth
  plt.imshow(image)
  cv2.waitKey(0)

