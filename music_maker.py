from Data import midi_io, data_parser, data_io
from Model import model_maker, model_io, model_trainer
import constants, math, os, random

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

def get_model_output(model, input):
    return

#gets the start of a random song
def get_initial_input():
    songs_list = os.listdir(constants.MIDI_INPUT_LOCATION)
    # initial_song = songs_list[0]
    initial_song = songs_list[random.randint(0, len(songs_list) - 1)]
    # initial_song = 'MIDI-UNPROCESSED_21-22_R1_2014_MID--AUDIO_22_R1_2014_wav--1.midi'
    print(f"starting with song: {initial_song}")
    
    piano_roll = midi_io.load_midi_as_piano_roll(initial_song)
    data_x, data_y = data_parser.piano_roll_to_training_data(piano_roll)
    return data_x[0]

# time of wanted song to number of sequences
def get_number_of_sequences(length_of_output):
    time_per_time_slice = constants.TIME_OF_TIME_SLICE
    time_slice_per_sequence = constants.INPUT_SEQUENCE_LENGTH
    
    seq_time = time_per_time_slice * time_slice_per_sequence
    return int(math.ceil(length_of_output / seq_time))


def piano_roll_from_network(model, length_in_sec):
    seq_count = get_number_of_sequences(length_in_sec)
    model_input = np.array([get_initial_input()])

    output = []
    output.append(model_input)
    for i in range(seq_count):
        model_input = model.predict([output[i]])
        output.append(model_input)

    return format_network_output(output[1:])

def format_network_output(nn_output):
    output = []
    for x in nn_output:
        output.append(x[0])
    return output

def make_song(model, length_in_sec):
    piano_roll = piano_roll_from_network(model, length_in_sec)
    midi_io.save_piano_roll_as_midi(piano_roll, length_in_sec)
    return


# data_io.save_all_midis_as_training_data()

# model = model_maker.create_nobi_model()
# model_trainer.train_model(model)
# model_io.save_model("jazz_mse_relu_nobi_96_30epoch_50seq.model", model)


# model = model_io.load_model_from_name("131702-jazz_huber_relu_nobi_96_30epoch_50seq.model")

# make_song(model, 30)

# good bots
# 230148-jazz_mse_tanh_192_10epoch.model
# 220732-jazz_mse_tanh_nobi_192_10epoch.model
# 131521-sqe_relu_96_30epoch.model
