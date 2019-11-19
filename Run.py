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

	# TODO: Recreate model with batch = 1 and stateful=True

	# Convert to single sample model
	weights = model.get_weights()
	single_item_model = create_model(batch_size=1)
	single_item_model.set_weights(weights)
	single_item_model.compile(compile_params)

	return model

def load_data(path):

	y, sr = librosa.load(path)
	y_part = y[0:511]
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
	
	print(model.predict(spect))

if __name__ == '__main__':
	main()