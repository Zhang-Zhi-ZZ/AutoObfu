import keras
import cv2
import numpy as np
from keras.applications import ResNet50
from keras.applications import InceptionV3
from keras.applications import Xception # TensorFlow ONLY
from keras.applications import VGG16
from keras.applications import VGG19
from keras.applications import imagenet_utils
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from keras.models import Sequential
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from keras import regularizers
from keras.callbacks import LearningRateScheduler
from keras.models import model_from_json
from keras.datasets import mnist, fashion_mnist,cifar10
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras import backend as K
from PIL import Image
from keras.models import model_from_json
import imagehash
import os
import glob
import csv




class predictor:
	def __init__(self):
		pass

	def general_pred(self):
		MODELS = {
			"vgg16": VGG16,
			"vgg19": VGG19,
			"inception": InceptionV3,
			"xception": Xception, # TensorFlow ONLY
			"resnet": ResNet50
		}

		inputShape = (224, 224)
		preprocess = imagenet_utils.preprocess_input

		Network = MODELS["vgg19"]
		model = Network(weights="imagenet")
		image = load_img("input.png", target_size=inputShape)
		image = img_to_array(image)
		image = np.expand_dims(image, axis=0)
		image = preprocess(image)
		preds = model.predict(image)
		P = imagenet_utils.decode_predictions(preds)

		result = []
		for (i, (imagenetID, label, prob)) in enumerate(P[0]):
			result.append("{}. {}: {:.2f}%".format(i + 1, label, prob * 100))
		return result


	def pred_blur(self):
		class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
				'dog', 'frog', 'horse', 'ship', 'truck']
		img = np.asarray(cv2.imread("Blurred.png"))
		img = np.asarray([img])
		img = img.astype('float32')
		img = img/255
		num_classes = 10

		#data augmentation
		datagen = ImageDataGenerator(
			rotation_range=15,
			width_shift_range=0.1,
			height_shift_range=0.1,
			horizontal_flip=True,
		)
		datagen.fit(img)
		#training
		batch_size = 64
		json_file = open('model_blur/model_blurred_n.json', 'r')
		loaded_model_json = json_file.read()
		json_file.close()
		loaded_model = model_from_json(loaded_model_json)
		# load weights into new model
		loaded_model.load_weights("model_blur/model_blurred_n.h5")
		opt_rms = keras.optimizers.rmsprop(lr=0.001,decay=1e-6)
		loaded_model.compile(loss=keras.losses.categorical_crossentropy,
			optimizer=opt_rms,
			metrics=['accuracy'])

		co = loaded_model.predict_classes(np.asarray([img[0]]))
		print(class_names[co[0]])
		return class_names[co[0]]

	def pred_encrypt(self):
		batch_size = 128
		num_classes = 10
		epochs = 30
		img = np.asarray(Image.open("Encrypted.png").convert('L'))
		print (img.shape)

		img = np.asarray([img])

		if K.image_data_format() == 'channels_first':
		  img = img.reshape(img.shape[0], 1, 28, 28)
		  input_shape = (1, 28, 28)
		else:
		  img = img.reshape(img.shape[0], 28, 28, 1)
		  input_shape = (28, 28, 1)
		img = img.astype('float32')
		img /= 255
		class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
			'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

		json_file = open('model_encryption/model_encrypted.json', 'r')
		loaded_model_json = json_file.read()
		json_file.close()
		loaded_model = model_from_json(loaded_model_json)
		# load weights into new model
		loaded_model.load_weights("model_encryption/model_encrypted.h5")
		loaded_model.compile(loss=keras.losses.categorical_crossentropy,
			optimizer=keras.optimizers.Adadelta(),
			metrics=['accuracy'])

		co = loaded_model.predict_classes(img)
		print (class_names[co[0]])
		return class_names[co[0]]

	def pred_fingerprint(self,fp):
		class_name = ['0','1','2','3','4','5','6','7','8','9']
		hashed = int(str(fp),16)
		scores = [] 
		with open('hash_code.csv', mode='r') as csv_file:
			for row in csv_file:
				sub_scores = []
				fingerprint = row.strip().split(",")
				for i in fingerprint:
					fin = int(i,16)
					h = self.hamming(hashed,fin)
					sub_scores.append(h)
				scores.append(sum(sub_scores)/len(sub_scores))
		print(class_name[scores.index(min(scores))])
		return class_name[scores.index(min(scores))]

	def hamming(self, h1, h2):
		h, d = 0, h1 ^ h2
		while d:
			h += 1
			d &= d - 1
		return h






