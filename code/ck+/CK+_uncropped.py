import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import keras
from keras.utils import np_utils
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, BatchNormalization
from keras.optimizers import SGD,RMSprop,Adam
from keras.preprocessing.image import ImageDataGenerator
import shutil
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def newfolder():
    if not os.path.exists('/Users/cameronduffy/Documents/Winter 2021/CISC 499/datasets/ck+/CK+_uncropped'):
        os.makedirs('/Users/cameronduffy/Documents/Winter 2021/CISC 499/datasets/ck+/CK+_uncropped')
        os.makedirs('/Users/cameronduffy/Documents/Winter 2021/CISC 499/datasets/ck+/CK+_uncropped/anger')
        os.makedirs('/Users/cameronduffy/Documents/Winter 2021/CISC 499/datasets/ck+/CK+_uncropped/contempt')
        os.makedirs('/Users/cameronduffy/Documents/Winter 2021/CISC 499/datasets/ck+/CK+_uncropped/disgust')
        os.makedirs('/Users/cameronduffy/Documents/Winter 2021/CISC 499/datasets/ck+/CK+_uncropped/fear')
        os.makedirs('/Users/cameronduffy/Documents/Winter 2021/CISC 499/datasets/ck+/CK+_uncropped/happy')
        os.makedirs('/Users/cameronduffy/Documents/Winter 2021/CISC 499/datasets/ck+/CK+_uncropped/sadness')
        os.makedirs('/Users/cameronduffy/Documents/Winter 2021/CISC 499/datasets/ck+/CK+_uncropped/surprise')
# newfolder()

# remove only the black bar from top of image and timestamp from bottom - nothing else.
def crop_images():
    images_path = "/Users/cameronduffy/Documents/Winter 2021/CISC 499/datasets/ck+/CK+_Complete"
    uncropped_path = '/Users/cameronduffy/Documents/Winter 2021/CISC 499/datasets/ck+/CK+_uncropped'
    images_dir_list = os.listdir(images_path)
    if '.DS_Store' in images_dir_list:
        images_dir_list.remove('.DS_Store')

    for i in images_dir_list:
        data_path = images_path + "/" + str(i) + "/"
        data_dir_list = os.listdir(data_path)

        count = 0
        for image in data_dir_list:
            im = os.path.join(data_path, str(image))
            im = Image.open(im)

            # Size of the image in pixels (size of orginal image) 
            width, height = im.size

            # Setting the points for cropped image -- cropping the black bar off the top and the timestamp off the bottom
            left = 0
            top = 12
            right = width
            bottom = height - 58
            
            im = im.crop((left, top, right, bottom)) 
            count = count + 1
            file_name = uncropped_path + "/" + str(i) + "/" + str(i) + "_cropped" + str(count) + '.jpg'
            if not os.path.exists(file_name):
                im.save(file_name)
# crop_images()

#Use seed function to generate the random values at the same
np.random.seed(10)

parent_dir = os.path.dirname(__file__)
# go up two directories
for i in range(2):
    parent_dir = os.path.dirname(parent_dir)

rel_path = '/datasets/ck+/CK+_uncropped'
data_path = parent_dir + rel_path

data_dir_list = os.listdir(data_path)
if '.DS_Store' in data_dir_list:
    data_dir_list.remove('.DS_Store')
# print('ls: ', data_dir_list)

# preprocessing the image data
img_data_list = []
labels = []
labeldic={'anger':0 ,'contempt':1,'disgust':2,'fear':3,'happy':4,'sadness':5,'surprise':6}   #The labels and its value for CK+
for dataset in data_dir_list:
    img_list = os.listdir(data_path + '/' + dataset)
    # print ('Loaded the images of dataset-' + '{}'.format(dataset))
    for img in img_list:
        impath1=data_path + '/' + dataset + '/' + img
        input_img = cv2.imread(impath1)                         #Read the images
        input_img = cv2.resize(input_img, (48, 48))             #Resize the input image
        img_data_list.append(input_img)
        labels.append(labeldic[dataset])               #The label of each image is the name of its folder


img_data = np.array(img_data_list)                            #Convert image to array
img_data = img_data.astype('float32')                         #Convert values to float
img_data = img_data / 255                                     #image normalization
img_data.shape
num_classes = 7   # CK+                                            #Emotions labels

num_of_samples = img_data.shape[0]                              #No of samples

Y = np_utils.to_categorical(labels, num_classes)
# print('Y:\n', Y)

#Shuffle the dataset
x,y = shuffle(img_data,Y, random_state=20)
# Split the dataset
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.15, random_state=10)

datagen = ImageDataGenerator(
    validation_split=0.176,
    rotation_range=0.1,
    horizontal_flip=True,
    # vertical_flip=True,
    width_shift_range=1.0,
    # height_shift_range=1.0
)
datagen.fit(x_train)

print(type(datagen))

input_shape=img_data[0].shape
print(input_shape)

model = Sequential()

#CNN model structure
#1st layer(one convolution process + max pooling )
model.add(Conv2D(32,(3, 3), activation='relu',  input_shape=input_shape, padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2) ))

#2nd layer (one convolution process + max pooling )
model.add(Conv2D(64,( 3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

#3rd layer (one convolution process + Batch normalization + max pooling )
model.add(Conv2D(128,( 3, 3), activation='relu'))
model.add(BatchNormalization())                          #Batch_normalization
model.add(MaxPooling2D(pool_size=(2, 2)))

#flatten
model.add(Flatten())

#Fullyconnected layers
#1st fullyconnected layer
model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dropout(0.5))
#2nd fullyconnected layer
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))

##output layer with softmax function
model.add(Dense(num_classes))
model.add(Activation('softmax'))
#---------------------------------------------------------

##Compile Model
model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=["accuracy"])
#-------------------------------------------------

##Show the summary of the model
# model.summary()
#------------------------------------------

# model_hist  = model.fit(x_train, y_train, batch_size=300, epochs=100, verbose=0,  validation_split=0.176)
history = model.fit(datagen.flow(x_train, y_train, batch_size=300), epochs=150, verbose=1)
print("rotation range:", datagen.rotation_range)
print('RR 0.1°, HF, HT')

#---------------------------------------------
## Evaluating the model##
train_score = model.evaluate(x_train, y_train, verbose=0)
print('Train loss:', train_score[0])
print('Train accuracy:', 100*train_score[1])

test_score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', test_score[0])
print('Test accuracy:', 100*test_score[1])

#print("history:",hist.history)
test_image = x_test[0:1]
print ("test_image shape", test_image.shape)
print("The training samples ", len(x_train),"The training ratio ",len(x_train)/(len(x_train)+len(x_test)))

print("The testing samples", len(x_test),"The testing ratio ",len(x_test)/(len(x_train)+len(x_test)))

# make graphs
# train_loss=history.history['loss']
# val_loss=history.history['val_loss']
# train_acc=history.history['acc']
# val_acc=history.history['val_acc']

# epochs = range(len(train_acc))

# plt.plot(epochs,train_loss,'r', label='train_loss')
# plt.plot(epochs,val_loss,'b', label='val_loss')
# plt.title('train_loss vs val_loss')
# plt.legend()
# plt.figure()
# plt.show()

# plt.plot(epochs,train_acc,'r', label='train_acc')
# plt.plot(epochs,val_acc,'b', label='val_acc')
# plt.title('train_acc vs val_acc')
# plt.legend()
# plt.figure()
# plt.show()

