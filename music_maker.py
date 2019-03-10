from Data import midi_io, data_parser
from Model import model_maker
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import constants

# (data_x, data_y), (validation_x, validation_y) = midi_io.load_all_midis_as_training_testing_data()
data_x, data_y = midi_io.load_all_midis_as_training_data()

(training_x, training_y), (validation_x, validation_y) = data_parser.split_training_data(data_x, data_y)
print(np.array(training_x).shape)
print(np.array(training_y).shape)
print(np.array(validation_x).shape)
print(np.array(validation_y).shape)

model = model_maker.create_model()


print(model.summary())
model.fit(x = data_x, y = data_y, batch_size=64)
