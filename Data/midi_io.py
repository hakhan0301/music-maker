from Data import data_parser 
import constants, os, time, numpy as np

midi_input_location = constants.MIDI_INPUT_LOCATION

def load_all_midis_as_piano_roll():
	piano_rolls = []
	path_list = os.listdir(midi_input_location)
	
	for file_name in path_list:
		try:
			piano_rolls.append(load_midi_as_piano_roll(file_name))
		except:
			print(f"error on file: {file_name}")
	return piano_rolls

def load_all_midis_as_training_testing_data():
	data_x, data_y = load_all_midis_as_training_data()
	return data_parser.split_training_data(data_x, data_y)

def load_all_midis_as_training_data():
	piano_rolls = load_all_midis_as_piano_roll()
	return data_parser.piano_roll_array_to_training_data(piano_rolls)

def load_midi_as_piano_roll(file_name):
	return data_parser.midi_to_piano_roll(midi_input_location + file_name)

def save_piano_roll_as_midi(file_name, piano_roll_data):
	# some very quick time formatting
	# goes from: 'Sat Mar  9 12:11:28 2019' to: 'Mar 9 12:12:41 2019'
	midi_name = time.asctime()[4:].replace('  ', ' ')
	print(midi_name)
	return