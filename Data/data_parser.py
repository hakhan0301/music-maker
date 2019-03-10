import mido, math, constants, numpy as np

time_of_slice_time = constants.TIME_OF_TIME_SLICE
max_note = constants.MAX_NOTE
min_note = constants.MIN_NOTE
notes_count = constants.NOTES_COUNT

sequence_length = constants.INPUT_SEQUENCE_LENGTH
validation_data_split = constants.VALIDATION_DATA_SPLIT
data_input_size = constants.NEURAL_INPUT_SIZE

class Midi_Data():
	def __init__(self, total_tick_count, total_time_slice_count, ticks_per_time_slice):
		self.TOTAL_TICK_COUNT = total_tick_count
		self.TOTAL_TIME_SLICE_COUNT = total_time_slice_count
		self.TICKS_PER_TIME_SLICE = ticks_per_time_slice

class Training_Data: 
	def __init__(self, input, output):
		self.input = input
		self.output = output

def piano_roll_to_midi(pianoroll_data):
	return

def midi_to_piano_roll(file_path):
	midi_file = mido.MidiFile(file_path)
	midi_data = get_midi_file_data(midi_file)

	total_tick_count = midi_data.TOTAL_TICK_COUNT
	#ime_slice_count = midi_data.TOTAL_TIME_SLICE_COUNT
	ticks_per_time_slice = midi_data.TICKS_PER_TIME_SLICE

	piano_roll_data = np.zeros((notes_count, total_tick_count), dtype = int)

	# stores last time each note was on
	# used to determine how long note was activated for
	state_of_notes = {}

	for track in midi_file.tracks:
		tick_count = 0

		for event in track:
			if event.type == 'note_on' and event.velocity > 0:
				tick_count += event.time
				time_slice_index = int(tick_count / ticks_per_time_slice)

				if event.note >= min_note and event.note <= max_note:
					note_index = event.note - min_note
					piano_roll_data[note_index][time_slice_index] = 1
					state_of_notes[note_index] = time_slice_index
			
			elif event.type == 'note_off' or (event.type == 'note_on' and event.velocity == 0):
				tick_count += event.time
				note_index = event.note - min_note
				time_slice_index = int(tick_count / ticks_per_time_slice)

				if note_index in state_of_notes:
					# last stored 'on' state of note to current 'off' state is set to 1
					# print("filling from", state_of_notes[note_index], " to ", time_slice_index)
					piano_roll_data[note_index][state_of_notes[note_index]:time_slice_index] = 1
					del state_of_notes[note_index]

	return piano_roll_data.T

def get_midi_file_data(midi_file):
	ticks_per_beat = midi_file.ticks_per_beat
	
	tempo_events = [x for t in midi_file.tracks for x in t if str(x.type) == 'set_tempo']
	tempo = mido.tempo2bpm(tempo_events[0].tempo)

	ticks_per_time_slice = float(ticks_per_beat * tempo * time_of_slice_time / 60)

	total_tick_count = 0
	for track in midi_file.tracks:
		ticks = 0

		for event in track:
			if event.type == 'note_on' or event.type == 'note_off' or event.type == 'end_of_track':
				ticks += event.time
		
		if ticks > total_tick_count:
			total_tick_count = ticks

	time_slice_count = int(math.ceil(total_tick_count / ticks_per_time_slice))

	return Midi_Data(total_tick_count, time_slice_count, ticks_per_time_slice)

#if piano_roll data goes to index 200 and sequence length is 50
#piano roll is split into indices ranging from __ to __
#x         :   y
#0 to 50   :   50 to 100
#50 to 100 :   100 to 150
#100 to 150:   150 to 200
def piano_roll_to_training_data(piano_roll):
	input = []
	output = []
	
	#removes elements from list so it can fit inputs and outputs
	psuedo_sequence = sequence_length - 1

	piano_roll = piano_roll[:len(piano_roll) - len(piano_roll) % (psuedo_sequence)]
	# piano_roll = piano_roll[:8]

	new_col = np.zeros((len(piano_roll), 1), dtype=int)
	piano_roll = np.append(piano_roll, new_col, 1)

	range_to_parse = len(piano_roll) / (psuedo_sequence) - 1

	# tag that tells neural net to start decoding
	end_tag = np.zeros((piano_roll[0].shape))
	end_tag[-1] = 1
	for i in range(int(range_to_parse)):
		input_start = i * psuedo_sequence 
		input_end = i * psuedo_sequence + psuedo_sequence

		output_start = i * psuedo_sequence + psuedo_sequence
		output_end = i * psuedo_sequence + psuedo_sequence * 2

		# print(input_start, input_end)
		# print(output_start, output_end)

		input_array = piano_roll[input_start:input_end]  
		output_array = piano_roll[output_start:output_end]

		input_array = np.vstack([input_array, end_tag])
		output_array = np.vstack([output_array, end_tag])

		input.append(input_array)
		output.append(output_array)

		# input.append(np.append(input_array, end_tag))
		# output.append(np.append(output_array, end_tag))
	
	return np.array(input), np.array(output)

def split_training_data(training_data_x, training_data_y):
	split_pos = len(training_data_x) - math.ceil(validation_data_split * len(training_data_x))

	validation_x = training_data_x[split_pos:]
	validation_y = training_data_y[split_pos:]

	training_data_x = training_data_x[:split_pos]
	training_data_y = training_data_x[:split_pos]

	return (training_data_x, training_data_y), (validation_x, validation_y)