from Data import midi_io, data_parser
from Model import model_maker
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

piano_roll_data = midi_io.load_all_midis_as_piano_roll()

training_data_x, training_data_y = data_parser.piano_roll_to_training_data(piano_roll_data[0])

model = model_maker.create_model()
# print(model.summary())

optimizer = tf.keras.optimizers.Adam(lr = 0.001, decay = 1e-6)
model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

x = np.array(training_data_x[0])
y = np.array(training_data_y[0])

print(model.input)
print(x.shape)
print(y.shape)

# model.fit(x=[x], y=y, epochs=2)
