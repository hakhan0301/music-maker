from Data import midi_io
from Model import model_maker
import tensorflow as tf

model = model_maker.create_model()
data = midi_io.load_all_midis_as_piano_roll()
model.compile(optimizer='adam',  loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(data[0][0], data[0][1], epochs=20)