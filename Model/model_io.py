from constants import MODEL_LOCATION
import time, os, tensorflow as tf

def save_model(file_name, model):
	model.save(MODEL_LOCATION + f"{time.asctime()[11:19].replace(':','')}-{file_name}")
	return

def load_model_from_index(index):
	file_list = os.listdir(MODEL_LOCATION)
	print(f"returning file: {file_list[index]}")
	return load_model_from_name(file_list[index])

def load_model_from_name(file_name):
	return tf.keras.models.load_model(MODEL_LOCATION + file_name)