import bz2
import os
import cv2
from urllib.request import urlopen
from align import AlignDlib
import numpy as np
from model import create_model
from keras.models import model_from_json

Path_Image = "test2.jpeg"
def load_image_alignment(path):
	alignment = AlignDlib('models/landmarks.dat')
	img = cv2.imread(path,1)
	# detect face and return bounding box
	bb = alignment.getLargestFaceBoundingBox(img)
	jc_aligned = alignment.align(96, img, bb, landmarkIndices=AlignDlib.OUTER_EYES_AND_NOSE)
	return jc_aligned

loaded_model = create_model()
loaded_model.load_weights("./weights/model.h5")
def predict_image(path):
	img = load_image_alignment(Path_Image)
	img = np.expand_dims(img, axis=0)
	a = np.argmax(loaded_model.predict(img))+1
	return a


img = cv2.imread(Path_Image,1)
alignment = AlignDlib('models/landmarks.dat')
bb = alignment.getLargestFaceBoundingBox(img)
cv2.rectangle(img,(bb.left(),bb.top()),(bb.right(),bb.right()+(bb.top()-bb.left())),(0,0,255),5)
cv2.imshow("anh thuoc nhom:"+ str(predict_image(Path_Image)),img)
cv2.waitKey(0)

