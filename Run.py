import librosa
import librosa.display

from keras.models import model_from_json

import matplotlib.pyplot as plt
import numpy as np

import os

import Data

def load_model(path):	
	# Model reconstruction from JSON file
	with open(os.path.join(path, 'model_architecture.json'), 'r') as f:
	    model = model_from_json(f.read())

	# Load weights into the new model
	model.load_weights(os.path.join(path, 'model_weights.h5'))

	# Convert to stateful model
	weights = model.get_weights()
	stateful_model = Model.build_model(weights=weights, stateful=True)

	return stateful_model

def load_data(path):

	N_SAMPLES = 5

	y, sr = librosa.load(path)
	y_part = y[0:N_SAMPLES*512-1]
	spect = Data.convert_audio_data(y_part, sr)
	spect = spect[np.newaxis, :, :]

	return spect

def main():
	# Load model
	model = load_model("models/crnn")
	print(model.summary())
	
	# Load test data
	spect = load_data("data/raw/music/fma_small/000/000002.mp3")
	print(spect.shape)
	batch_size = spect.shape[0]
	
	print(model.predict(spect, batch_size=batch_size))

if __name__ == '__main__':
	main()