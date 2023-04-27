#pip install facenet_keras

import os
os.chdir(r"C:\Users\Janarish\Desktop\Face_Detection")

from os import listdir
from os.path import isdir
from PIL import Image
from numpy import asarray
from mtcnn import MTCNN

from numpy import expand_dims

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC

from keras_facenet import FaceNet
 
# extract a single face from a given photograph
def extract_face(filename, required_size=(160, 160)):
	# load image from file
	image = Image.open(filename)
	# convert to RGB, if needed
	image = image.convert('RGB')
	# convert to array
	pixels = asarray(image)
	# create the detector, using default weights
	detector = MTCNN()
	# detect faces in the image
	results = detector.detect_faces(pixels)
	# extract the bounding box from the first face
	x1, y1, width, height = results[0]['box']
	# bug fix
	x1, y1 = abs(x1), abs(y1)
	x2, y2 = x1 + width, y1 + height
	# extract the face
	face = pixels[y1:y2, x1:x2]
	# resize pixels to the model size
	image = Image.fromarray(face)
	image = image.resize(required_size)
	face_array = asarray(image)
	return face_array
 
# load images and extract faces for all images in a directory
def load_faces(directory):
	faces = list()
	# enumerate files
	for filename in listdir(directory):
		# path
		path = directory + filename
		# get face
		face = extract_face(path)
		# store
		faces.append(face)
	return faces
 
# load a dataset that contains one subdir for each class that in turn contains images
def load_dataset(directory):
	X, y = list(), list()
	# enumerate folders, on per class
	for subdir in listdir(directory):
		# path
		path = directory + subdir + '/'
		# skip any files that might be in the dir
		if not isdir(path):
			continue
		# load all faces in the subdirectory
		faces = load_faces(path)
		# create labels
		labels = [subdir for _ in range(len(faces))]
		# summarize progress
		print('>loaded %d examples for class: %s' % (len(faces), subdir))
		# store
		X.extend(faces)
		y.extend(labels)
	return asarray(X), asarray(y)
 
# load train dataset
trainX, trainy = load_dataset('data/train/')

# get the face embedding for one face
def get_embedding(model, face_pixels):
	# scale pixel values
	face_pixels = face_pixels.astype('float32')
	samples = expand_dims(face_pixels, axis=0)
	# make prediction to get embedding
	yhat = model.embeddings(samples)
	return yhat[0]

# load the facenet model
model = FaceNet();
print('Loaded Model')
# convert each face in the train set to an embedding
newTrainX = list()
for face_pixels in trainX:
	embedding = get_embedding(model, face_pixels)
	newTrainX.append(embedding)
newTrainX = asarray(newTrainX)
print(newTrainX.shape)

# normalize input vectors
in_encoder = Normalizer(norm='l2')
trainX = in_encoder.transform(newTrainX)
# label encode targets
out_encoder = LabelEncoder()
out_encoder.fit(trainy)

#Save Encoder
import pickle
with open('encoder/out_encoder.pkl', 'wb') as f:
    pickle.dump(out_encoder, f)
trainy = out_encoder.transform(trainy)

# fit model
model = SVC(kernel='linear', probability=True)
model.fit(trainX, trainy)
trained_model = model

#Save model
import pickle
with open('model/trained_model.pkl', 'wb') as f:
    pickle.dump(trained_model, f)