from Data import midi_io, data_parser, data_io
from Model import model_maker
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import constants

piano_roll = midi_io.load_midi_as_piano_roll("_struggle.mid")
# training_data = data_parser.piano_roll_to_training_data(piano_roll)

print(np.sum(piano_roll[-500000:]))
plt.imshow(piano_roll[0:])
plt.show()