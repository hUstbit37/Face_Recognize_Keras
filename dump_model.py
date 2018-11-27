import numpy as np
import os
import cv2
import random 
from sklearn.metrics import f1_score, accuracy_score
from model import create_model
from keras import backend as K
from keras.models import Model
from keras.models import model_from_json
import keras
PATH_IMAGE_CROP = "images_crop/"
data_feature = []
data_label= []
for i in os.listdir(PATH_IMAGE_CROP):
	for j in os.listdir(PATH_IMAGE_CROP + i + "/"):
		img = cv2.imread(PATH_IMAGE_CROP + i + "/"+j,1)
		data_feature.append(img)
		data_label.append(int(i)-1)
data_feature = np.asarray(data_feature)
data_label   = np.asarray(data_label)
data_shuffle = np.arange(data_feature.shape[0])
random.shuffle(data_shuffle)
training_samples = int(data_feature.shape[0] * 0.9)
train_idx = data_shuffle[:training_samples]
test_idx = data_shuffle[training_samples:]
#data_training
training_feature = data_feature[train_idx]
#print(training_feature.shape)
training_label   = data_label[train_idx]
training_label = keras.utils.to_categorical(training_label, 10)
#print(training_label.shape)
#data_testing
testing_feature = data_feature[test_idx]
testing_label   = data_label[test_idx]
testing_label= keras.utils.to_categorical(testing_label, 10)
model = create_model()
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])
model.fit(training_feature, training_label,
          batch_size=10,
          epochs=25,
          verbose=1,
          validation_data=(testing_feature, testing_label)) 
score = model.evaluate(testing_feature, testing_label, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])  

model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")                     

