import mido, math, constants, numpy as np

time_of_slice_time = constants.TIME_OF_TIME_SLICE
max_note = constants.MAX_NOTE
min_note = constants.MIN_NOTE
notes_count = constants.NOTES_COUNT
value_of_note_on = constants.NOTE_ACTIVE_VALUE

sequence_length = constants.INPUT_SEQUENCE_LENGTH
validation_data_split = constants.VALIDATION_DATA_SPLIT
data_input_size = constants.NEURAL_INPUT_SIZE

def process_piano_roll(preprocessed_piano_roll_data):
	piano_roll_data = []
	for x in preprocessed_piano_roll_data:
		for y in x:
			piano_roll_data.append(y)

	output = []
	for x in piano_roll_data:
		x_1 = []
		for y in x:
			if (y > constants.NOTE_CUTOFF):
				y = 1
			else:
				y = 0
			x_1.append(y)
		output.append(x_1)
	output = np.array(output)
	return output


#copied from microsoft midi converter
def piano_roll_to_midi(preprocessed_piano_roll_data, song_length):
	print('creating midi file')
	piano_roll_data = process_piano_roll(preprocessed_piano_roll_data)
	ticks_per_time_slice = 1
	tempo = 1 / time_of_slice_time
	resolution = 60 * ticks_per_time_slice / (tempo * time_of_slice_time)

	mid = mido.MidiFile(ticks_per_beat = int(resolution))
	track = mido.MidiTrack()
	mid.tracks.append(track)
	track.append(mido.MetaMessage('set_tempo', tempo = int(60000000.0 / tempo), time = 0))

	current_state = np.zeros(notes_count)

	index_of_last_event = 0
	for slice_index, time_slice in enumerate(np.concatenate((piano_roll_data, np.zeros((1, notes_count))), axis = 0)):
		note_changes = time_slice - current_state
		
		for note_idx, note in enumerate(note_changes):
			if note == 1:
				note_event = mido.Message('note_on', time = (slice_index - index_of_last_event)*ticks_per_time_slice, velocity = 65, note = note_idx + min_note )
				track.append(note_event)
				index_of_last_event = slice_index
			elif note == -1:
				note_event = mido.Message('note_off', time = (slice_index - index_of_last_event)*ticks_per_time_slice, velocity = 65, note = note_idx + min_note )
				track.append(note_event)
				index_of_last_event = slice_index

		current_state = time_slice

	song_length_in_ticks = (song_length / time_of_slice_time) * ticks_per_time_slice
	song_length_in_ticks = int(song_length_in_ticks)
	eot = mido.MetaMessage('end_of_track', time = song_length_in_ticks)
	track.append(eot)
	
	return mid

#copied from microsoft midi converter
def midi_to_piano_roll(file_path):
	midi_data = mido.MidiFile(file_path)
	resolution = midi_data.ticks_per_beat
	set_tempo_events = [x for t in midi_data.tracks for x in t if str(x.type) == 'set_tempo']
	
	tempo = 60000000.0/set_tempo_events[0].tempo
	ticks_per_time_slice = 1.0 * (resolution * tempo * time_of_slice_time)/60 
	
	total_ticks = 0
	for t in midi_data.tracks:
		sum_ticks = 0
		for e in t:
			if str(e.type) == 'note_on' or str(e.type) == 'note_off' or str(e.type) == 'end_of_track':
				sum_ticks += e.time
				
		if sum_ticks > total_ticks:
			total_ticks = sum_ticks

	time_slices = int(math.ceil(total_ticks / ticks_per_time_slice))

	piano_roll = np.zeros((notes_count, time_slices), dtype =int)

	note_states = {}
	for track in midi_data.tracks:
		total_ticks = 0
		for event in track:
			if str(event.type) == 'note_on' and event.velocity > 0:
				total_ticks += event.time
				time_slice_idx = int(total_ticks / ticks_per_time_slice )

				if event.note <= max_note and event.note >= min_note: 
					note_idx = event.note - min_note
					piano_roll[note_idx][time_slice_idx] = value_of_note_on
					note_states[note_idx] = time_slice_idx

			elif str(event.type) == 'note_off' or ( str(event.type) == 'note_on' and event.velocity == 0 ):
				note_idx = event.note - min_note
				total_ticks += event.time
				time_slice_idx = int(total_ticks /ticks_per_time_slice )

				if note_idx in note_states:	
					last_time_slice_index = note_states[note_idx]
					piano_roll[note_idx][last_time_slice_index:time_slice_idx] = value_of_note_on
					del note_states[note_idx]
	return piano_roll.T

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
	psuedo_sequence = sequence_length

	piano_roll = piano_roll[:len(piano_roll) - len(piano_roll) % (psuedo_sequence)]
	# piano_roll = piano_roll[:8]

	# new_col = np.zeros((len(piano_roll), 1), dtype=bool)
	# piano_roll = np.append(piano_roll, new_col, 1)

	range_to_parse = len(piano_roll) / (psuedo_sequence) - 1

	# tag that tells neural net to start decoding
	# end_tag = np.zeros((piano_roll[0].shape))
	# end_tag[-1] = 1

	for i in range(int(range_to_parse)):
		input_start = i * psuedo_sequence 
		input_end = i * psuedo_sequence + psuedo_sequence

		output_start = i * psuedo_sequence + psuedo_sequence
		output_end = i * psuedo_sequence + psuedo_sequence * 2

		# print(input_start, input_end)
		# print(output_start, output_end)

		input_array = piano_roll[input_start:input_end]  
		output_array = piano_roll[output_start:output_end]

		# input_array = np.vstack([input_array, end_tag])
		# output_array = np.vstack([output_array, end_tag])

		input.append(input_array)
		output.append(output_array)

		# input.append(np.append(input_array, end_tag))
		# output.append(np.append(output_array, end_tag))
	
	return np.array(input), np.array(output)

def piano_roll_array_to_training_data(piano_rolls):
	training_data_x = []
	training_data_y = []
	for piano_roll in piano_rolls:
		data_x, data_y = piano_roll_to_training_data(piano_roll)
		for x in data_x:
			training_data_x.append(x)
		for y in data_y:
			training_data_y.append(y)
	
	return np.array(training_data_x), np.array(training_data_y)


def split_training_data(training_data_x, training_data_y):
	split_pos = len(training_data_x) - math.ceil(validation_data_split * len(training_data_x))

	validation_x = training_data_x[split_pos:]
	validation_y = training_data_y[split_pos:]

	training_data_x = training_data_x[:split_pos]
	training_data_y = training_data_x[:split_pos]

	return (training_data_x, training_data_y), (validation_x, validation_y)