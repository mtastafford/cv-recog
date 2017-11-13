# import the necessary packages
import numpy as np
import cv2
from imutils.face_utils import FaceAligner
from imutils.face_utils import rect_to_bb
import dlib 
import imutils

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor and the face aligner
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
fa = FaceAligner(predictor, desiredFaceWidth=256)

cap = cv2.VideoCapture(0)
count = 0
name = raw_input("Enter your name....")

while True:
	ret, img = cap.read()
	img = imutils.resize(img, width=800)
	laplacian = cv2.Laplacian(img,cv2.CV_64F)
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	cv2.imshow("Input", img)
	rects = detector(gray, 2)
	#loop over the face detections
	print('Faces found!', len(rects))
	for rect in rects:
		#extract the ROI of the *original* face, then align the face
		#using facial landmarks
		(x, y, w, h) = rect_to_bb(rect)
		faceOrig = imutils.resize(img[y:y+h, x:x+w], width = 256)
		faceAligned = fa.align(img, gray, rect)
		saver = cv2.cvtColor(faceAligned, cv2.COLOR_BGR2GRAY)
		cv2.imwrite(name + str(count) + '.png', saver)
#		cv2.imshow("Original", faceOrig)
#		cv2.imshow("Aligned", faceAligned)
		count += 1
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()
