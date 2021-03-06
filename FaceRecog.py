# import the necessary packages
from imutils.face_utils import FaceAligner
from imutils.face_utils import rect_to_bb
import cv2, os
import numpy as np
import sys
import dlib
import imutils

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
cap = cv2.VideoCapture(0)
subjects = [' ', 'Caitlyn', 'Mark']

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
fa = FaceAligner(predictor, desiredFaceWidth=256)

#function to detect face using OpenCV
def detect_face(img):
	#convert the test image to gray scale as opencv face detector expects gray images
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	#load OpenCV face detector, I am using LBP which is fast
	#there is also a more accurate but slow: Haar classifier
#	face_cascade = cv2.CascadeClassifier('lbpcascade_frontalface.xml')
	#let's detect multiscale images(some images may be closer to camera than others)
	#result is a list of faces
#	faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=6);
	rects = detector(gray, 2)
	#if no faces are detected then return original img
	if (len(rects) == 0):
		return None, None
	#under the assumption that there will be only one face,
	#extract the face area
	print rects[0]
	for rect in rects:
		(x, y, w, h) = rect_to_bb(rect)
	#return only the face part of the image
		return gray[y:y+w, x:x+h], rect_to_bb(rect)

#this function will read all persons' training images, detect face from each image
#and will return two lists of exactly same size, one list 
#of faces and another list of labels for each face

def prepare_training_data(data_folder_path):
	#------STEP-1--------
	#get the directories (one directory for each subject) in data folder
	dirs = os.listdir(data_folder_path)
	#list to hold all subject faces
	faces = []
	#list to hold labels for all subjects
	labels = []
	#let's go through each directory and read images within it
	for dir_name in dirs:
	#our subject directories start with letter 's' so
	#ignore any non-relevant directories if any
		if not dir_name.startswith("s"):
			continue;
		#------STEP-2--------
		#extract label number of subject from dir_name
		#format of dir name = slabel
		#, so removing letter 's' from dir_name will give us label
		label = int(dir_name.replace("s", ""))
		#build path of directory containing images for current subject subject
		#sample subject_dir_path = "training-data/s1"
		subject_dir_path = data_folder_path + "/" + dir_name
		#get the images names that are inside the given subject directory
		subject_images_names = os.listdir(subject_dir_path)
		#------STEP-3--------
		#go through each image name, read image, 
		#detect face and add face to list of faces
		for image_name in subject_images_names:
			#ignore system files like .DS_Store
			if image_name.startswith("."):
				continue;
			#build image path
			#sample image path = training-data/s1/1.pgm
			image_path = subject_dir_path + "/" + image_name
			#read image
			image = cv2.imread(image_path)
			#display an image window to show the image 
			cv2.imshow("Training on image...", image)
			cv2.waitKey(100)
			#detect face
			face, rect = detect_face(image)
			#------STEP-4--------
			#for the purpose of this tutorial
			#we will ignore faces that are not detected
			if face is not None:
			#add face to list of faces
				faces.append(face)
				#add label for this face
				labels.append(label)
				cv2.destroyAllWindows()
	cv2.waitKey(1)
	cv2.destroyAllWindows()
 	return faces, labels
faces, labels = prepare_training_data("training-data")
print("Data prepared")
#print total faces and labels
print("Total faces: ", len(faces))
print("Total labels: ", len(labels))

#create our LBPH face recognizer 
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
 
#or use EigenFaceRecognizer by replacing above line with 
#face_recognizer = cv2.face.EigenFaceRecognizer_create()
 
#or use FisherFaceRecognizer by replacing above line with 
#face_recognizer = cv2.face.createFisherFaceRecognizer()

#train our face recognizer of our training faces
face_recognizer.train(faces, np.array(labels))

def draw_text(img, text, x, y):
	cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_PLAIN, 0.5, (0, 0, 255), 1)

def draw_rectangle(img, rect):
	(x, y, w, h) = rect
	cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

#this function recognizes the person in image passed
#and draws a rectangle around detected face with name of the 
#subject
def predict(test_img):
	#make a copy of the image as we don't want to change original image
	img = test_img.copy()
	#detect face from the image
	face, rect = detect_face(img)
	#predict the image using our face recognizer 
	label = face_recognizer.predict(face)
	#get name of respective label returned by face recognizer
	label_text = str(subjects[label[0]])+", "+str(label[1])+"%"
	#draw a rectangle around face detected
#	draw_rectangle(img, rect)
	#draw name of predicted person
#	draw_text(img, label_text, rect[0], rect[1]-5)
	return img, label_text

while True:
	ret, frame = cap.read()
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	print("Looking for Faces")
	rects = detector(gray, 2)
	print("Trying to Recognize Faces...")
	#load test images
	for rect in rects:
		(x, y, w, h) = rect_to_bb(rect)
		faceOrig = imutils.resize(frame[y:y+h, x:x+w], width = 256)
		faceAligned = fa.align(frame, gray, rect)
		cv2.imshow("Checking", faceAligned)
		test_img1 = frame
		#perform a prediction
		predicted_img1, guess = predict(test_img1)
		print "Found: " + guess
		#display both images
#		cv2.imshow(subjects[1], predicted_img1)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()
