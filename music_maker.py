from Data import midi_io, data_parser, data_io
from Model import model_maker
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import constants




# piano_roll = midi_io.load_midi_as_piano_roll("_struggle.mid")
# plt.imshow(piano_roll[:constants.INPUT_SEQUENCE_LENGTH])
# plt.show()


data_io.save_all_midis_as_training_data()


# model = model_maker.create_model()
# # print(model.summary())
# data_x, data_y = data_io.load_training_data_file()
# print(np.array(data_x).shape)
# plt.imshow(data_x[50])
# plt.show()

# (data_x, data_y), (validation_x, valdation_y) = data_parser.split_training_data(data_x, data_y)

# model.fit(x=data_x, y=data_y, batch_size=64, validation_data=(validation_x, validation_y))
