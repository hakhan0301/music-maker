import tensorflow as tf
from Data import midi_io, data_parser, data_io
import constants

batches = constants.BATCHES
epochs = constants.EPOCHS

def train_model(model):
	data_x, data_y = data_io.load_training_data_file()
	(data_x, data_y), validation_data = data_parser.split_training_data(data_x, data_y)

	return model.fit(x=data_x, y=data_y, batch_size=batches, validation_data=validation_data, epochs=epochs)