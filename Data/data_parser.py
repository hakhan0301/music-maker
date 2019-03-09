import mido, math, constants, numpy as np

time_of_slice_time = constants.TIME_OF_TIME_SLICE
max_note = constants.MAX_NOTE
min_note = constants.MIN_NOTE
notes_count = constants.NOTES_COUNT

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
	time_slice_count = midi_data.TOTAL_TIME_SLICE_COUNT
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
					# last stored 'on' state of note to current 'off' is set to 1
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