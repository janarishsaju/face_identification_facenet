# =============================================================================
# Predict
# =============================================================================
import os
os.chdir(r"C:\Users\Janarish\Desktop\Face_Detection")

import pickle
import pandas as pd

# function for face detection with mtcnn
from PIL import Image
from numpy import asarray
from mtcnn.mtcnn import MTCNN
from sklearn.preprocessing import Normalizer
from os import listdir
from os.path import isdir
from matplotlib import pyplot

#Calculate a face embedding for each face in the dataset using facenet
from numpy import expand_dims

from keras_facenet import FaceNet

#Load Facenet model
model = FaceNet();

#Load model
with open('model/trained_model.pkl', 'rb') as f:
    trained_model = pickle.load(f)

#Load Encoder
with open('encoder/out_encoder.pkl', 'rb') as f:
    out_encoder = pickle.load(f)

#Extract a single face from a given photograph
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
	return asarray(X)
  
# load test dataset
predictX_faces = load_dataset('data/unknown/')

# get the face embedding for one face
def get_embedding(model, face_pixels):
	# scale pixel values
	face_pixels = face_pixels.astype('float32')
	samples = expand_dims(face_pixels, axis=0)
	# make prediction to get embedding
	yhat = model.embeddings(samples)
	return yhat[0]

newTestX = list()
for face_pixels in predictX_faces:
	embedding = get_embedding(model, face_pixels)
	newTestX.append(embedding)
newTestX = asarray(newTestX)
print(newTestX.shape)

in_encoder = Normalizer(norm='l2')
predictX = in_encoder.transform(newTestX)

#Create dataframe
df = pd.DataFrame(columns=['Image_Name','Detected_Person', 'Confidence_Score'])

for i in range(len(predictX_faces)):
    random_face_pixels = predictX_faces[i]
    random_face_emb = predictX[i]
    samples = expand_dims(random_face_emb, axis=0)
    yhat_class = trained_model.predict(samples)
    yhat_prob = trained_model.predict_proba(samples)
    # get name
    class_index = yhat_class[0]
    class_probability = yhat_prob[0,class_index] * 100
    predict_names = out_encoder.inverse_transform(yhat_class)
    print('Predicted: %s (%.3f)' % (predict_names[0], class_probability))
    df.at[i, "Image_Name"] = os.listdir("data/unknown/unknown")[i]
    df.at[i, "Detected_Person"] = predict_names[0]
    df.at[i, "Confidence_Score"] = class_probability
    # plot
    pyplot.imshow(random_face_pixels)
    title = '%s (%.3f)' % (predict_names[0], class_probability)
    pyplot.title(title)
    pyplot.show()

#Save prediction as dataframe
df.to_csv("predictions/predicted.csv", index=False)
