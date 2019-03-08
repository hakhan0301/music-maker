import mido, math, matplotlib.pyplot as plt
import numpy as np

class Midi_Data():
	def __init__(self, total_tick_count, total_time_slice_count, ticks_per_time_slice):
		self.total_tick_count = total_tick_count
		self.total_time_slice_count = total_time_slice_count
		self.ticks_per_time_slice = ticks_per_time_slice

TIME_OF_TIME_SLICE = .02
MAX_NOTE = 81
MIN_NOTE = 33
NOTES_DIMENSION = MAX_NOTE - MIN_NOTE + 1

def get_midi_file_data(midi_file):
	ticks_per_beat = midi_file.ticks_per_beat
	
	tempo_events = [x for t in midi_file.tracks for x in t if str(x.type) == 'set_tempo']
	tempo = mido.tempo2bpm(tempo_events[0].tempo)

	ticks_per_time_slice = float(ticks_per_beat * tempo * TIME_OF_TIME_SLICE / 60)

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

def midi_to_piano_roll(file_path):
	midi_file = mido.MidiFile(file_path)
	midi_data = get_midi_file_data(midi_file)

	total_tick_count = midi_data.total_tick_count
	time_slice_count = midi_data.total_time_slice_count
	ticks_per_time_slice = midi_data.ticks_per_time_slice

	piano_roll_data = np.zeros((NOTES_DIMENSION, total_tick_count), dtype = int)

	# stores last time each note was on
	# used to determine how long note was activated for
	state_of_notes = {}
	for track in midi_file.tracks:
		tick_count = 0

		for event in track:
			if event.type == 'note_on' and event.velocity > 0:
				tick_count += event.time
				time_slice_index = int(tick_count / ticks_per_time_slice)

				if event.note >= MIN_NOTE and event.note <= MAX_NOTE:
					note_index = event.note - MIN_NOTE
					piano_roll_data[note_index][time_slice_index] = 1
					state_of_notes[note_index] = time_slice_index
			
			elif event.type == 'note_off' or (event.type == 'note_on' and event.velocity == 0):
				tick_count += event.time
				note_index = event.note - MIN_NOTE
				time_slice_index = int(tick_count / ticks_per_time_slice)

				if note_index in state_of_notes:
					# last stored 'on' state of note to current 'off' is set to 1
					# print("filling from", state_of_notes[note_index], " to ", time_slice_index)
					piano_roll_data[note_index][state_of_notes[note_index]:time_slice_index] = 1
					del state_of_notes[note_index]

	return piano_roll_data.T


piano_roll = midi_to_piano_roll('C:\Projects\Machine Learning\Music Maker - Week 1\Data\Pre-Processed\PKMN_X_Y_Route_8_By_Incinium.mid')

plt.imshow(piano_roll[0:200])
plt.show()