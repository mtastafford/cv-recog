# import the necessary packages
from imutils.face_utils import FaceAligner
from imutils.face_utils import rect_to_bb
import argparse
import imutils
import dlib
import cv2, os
import numpy as np
import sys

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
torso_cascade = cv2.CascadeClassifier('haarcascade_upperbody.xml')
cap = cv2.VideoCapture(0)
count = 0
name = 'Mark'
subjects = [" ", "Caitlyn", "Mark"]

while True:
	ret, img = cap.read()
	laplacian = cv2.Laplacian(img,cv2.CV_64F)
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#	edges = cv2.Canny(img, 75, 100)
	faces = face_cascade.detectMultiScale(gray, 1.5, 5)
	print('Faces found: ', len(faces))
	for (x,y,w,h) in faces:
		cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),1)
		roi_gray = gray[y:y+h, x:x+w]
		roi_color = img[y:y+h, x:x+w]
		eyes = eye_cascade.detectMultiScale(roi_gray)
		for (ex,ey,ew,eh) in eyes:
			cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),1)
		if count % 5 == 0:
			cropped = gray[y:y+h,x:x+w]
			cv2.imwrite(name + str(count) + '.png', cropped)
		count += 1
	cv2.imshow('frame', img)
#	cv2.imshow('gray', gray)
#	cv2.imshow('Laplacian', laplacian)
#       cv2.imshow('x', sobelx)
#       cv2.imshow('y', sobely)
#	cv2.imshow('edges', edges)

	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()
