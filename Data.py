import librosa
import matplotlib.pyplot as plt
import numpy as np

import os
from glob import glob

counters = {"train": 0, "validation": 0}

def read_and_convert_file(path):
	#dur = librosa.core.get_duration(filename=path)
	y, sr = librosa.load(path)

	return convert_audio_data(y, sr)

def convert_audio_data(y, sr):
	# Calculate Mel-spectrogram with db loudness
	spect = librosa.feature.melspectrogram(y, sr, n_fft=2048, hop_length=512)
	spect = librosa.power_to_db(spect, ref=np.max)

	# Normalize
	spect -= np.min(spect)
	spect /= np.max(spect)
	spect = spect[:, 0:1024]

	return spect.T

def save_data(spect, save_root, label, validation_data_ratio=0.3):

	# Setup path
	rand = np.random.uniform()
	if rand < validation_data_ratio:
		data_group = "validation"
	else:
		data_group = "train"
	

	save_path = os.path.join(save_root, data_group, label, "%08d.npy" % counters[data_group])
	counters[data_group] += 1

	# Write image
	np.save(save_path, spect)


def read_and_convert_raw_data(read_root, save_root, label):

	counters["train"] = 0
	counters["validation"] = 0

	file_path_list = [y for x in os.walk(read_root) for y in glob(os.path.join(x[0], '*.mp3'))]

	print("Converting %d files..." % len(file_path_list))
	min_len = 1e9
	for i, path in enumerate(file_path_list[1:950]):

		spect = read_and_convert_file(path)

		min_len = min(min_len, spect.shape[1])
		save_data(spect, save_root, label)

		print("%d " % i, end="")


	print(min_len)




if __name__ == '__main__':
	labels = ["music", "talk"]

	for label in labels:
		read_and_convert_raw_data("data\\raw\\%s" % label, "data", "%s" % label)