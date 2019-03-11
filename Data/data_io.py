import constants, pickle, os, time, numpy as np
from Data import midi_io, data_parser

midi_input_location = constants.MIDI_INPUT_LOCATION
data_output_location = constants.DATA_OUTPUT_LOCATION
pickle_file_name = constants.PICKLE_FILE_NAME

def save_all_midis_as_training_data():
	path_list = os.listdir(midi_input_location)

	file_handler = open(data_output_location + pickle_file_name, 'wb')

	i = 0
	for file_name in path_list:
		i += 1
		if(i % 10 == 0):
			print(f"on file {i} out of {len(path_list)}")


		try:
			data = midi_io.load_midi_as_piano_roll(file_name)
			data = data_parser.piano_roll_to_training_data(data)

			pickle.dump(data, file_handler)
		except:
			print(f"error on file: {file_name}")
	
	file_handler.close()

def load_training_data_file():
	return pickle.load(open(data_output_location + pickle_file_name, "rb"))
	
