import bz2
import os
import cv2
from urllib.request import urlopen
from align import AlignDlib
import numpy as np

# Initialize the OpenFace face alignment utility
PATH_IMAGE      = "./images/"
PATH_IMAGE_CROP = "./images_crop/"
def load_image_alignment(path):
	alignment = AlignDlib('models/landmarks.dat')
	img = cv2.imread(path,1)
	# detect face and return bounding box
	bb = alignment.getLargestFaceBoundingBox(img)
	jc_aligned = alignment.align(96, img, bb, landmarkIndices=AlignDlib.OUTER_EYES_AND_NOSE)
	return jc_aligned

# load data
def load_data(PATH_IMAGE,PATH_IMAGE_CROP):
	for i in os.listdir(PATH_IMAGE):
		for j in os.listdir(PATH_IMAGE+i+"/"):
			img = load_image_alignment(PATH_IMAGE+i+"/"+j)
			cv2.imwrite(PATH_IMAGE_CROP+i+"/"+j,img)

load_data(PATH_IMAGE,PATH_IMAGE_CROP)

