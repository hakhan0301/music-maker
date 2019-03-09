from Data import data_parser 
import constants, os, time

midi_input_location = constants.MIDI_INPUT_LOCATION
midi_output_location = constants.MIDI_OUTPUT_LOCATION

def load_all_midis_as_piano_roll():
	piano_rolls = []
	for file_name in os.listdir(midi_input_location):
		piano_rolls.append(load_midi_as_piano_roll(file_name))
	return piano_rolls

def load_midi_as_piano_roll(file_name):
	return data_parser.midi_to_piano_roll(midi_input_location + file_name)

def save_piano_roll_as_midi(file_name, piano_roll_data):
	# some very quick time formatting
	# goes from: 'Sat Mar  9 12:11:28 2019' to: 'Mar 9 12:12:41 2019'
	midi_name = time.asctime()[4:].replace('  ', ' ')
	print(midi_name)
	return