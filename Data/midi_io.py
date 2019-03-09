from Data import data_parser 
import constants

midi_input_location = constants.MIDI_INPUT_LOCATION
midi_output_location = constants.MIDI_OUTPUT_LOCATION

def load_midi_as_piano_roll(file_name):
	return data_parser.midi_to_piano_roll(midi_input_location + file_name)

def save_piano_roll_as_midi(file_name, piano_roll_data):
	midi_file = data_parser.piano_roll_to_midi(piano_roll_data)
	return
