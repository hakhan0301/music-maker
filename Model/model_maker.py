import constants
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, RepeatVector, BatchNormalization, TimeDistributed, Bidirectional, Activation, CuDNNLSTM
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Conv2DTranspose, Reshape

input_shape = (constants.INPUT_SEQUENCE_LENGTH, constants.NEURAL_INPUT_SIZE)
output_shape = constants.NEURAL_INPUT_SIZE
sequence_length = constants.INPUT_SEQUENCE_LENGTH

lr = constants.LEARNING_RATE
decay = constants.DECAY_RATE
loss = constants.TRAINING_LOSS
metrics = constants.TRAINING_METRICS
optimizer = tf.keras.optimizers.Adam(lr=lr, decay=decay)
activation = constants.ACTIVATION
layer_size = constants.LAYER_SIZE

conv_kernal_height = constants.CONV_KERNAL_HEIGHT


def create_conv_model():
	print(input_shape)
	model = Sequential()
	
	model.add(Reshape((constants.INPUT_SEQUENCE_LENGTH, constants.NEURAL_INPUT_SIZE, 1), input_shape=input_shape))
	model.add(Conv2D(12, kernel_size=(conv_kernal_height, constants.NEURAL_INPUT_SIZE), strides=(1, 1),
					 input_shape=input_shape))
	model.add(MaxPooling2D(1, 2))
	model.add(BatchNormalization())
	model.add(Activation("relu"))

	model.add(Flatten())
	model.add(Dense(32))
	model.add(BatchNormalization())
	model.add(Activation("relu"))

	model.add(Reshape((4, 4, 2)))
	model.add(Dropout(.2))

	# model.add(Conv2D(64, kernel_size=(2, 2)))
	# model.add(MaxPooling2D(2, 2))
	# model.add(BatchNormalization())
	# model.add(Activation("relu"))

	# model.add(Conv2DTranspose(64, kernel_size=(2, 2)))
	# model.add(BatchNormalization())
	# model.add(Activation("relu"))

	model.add(Conv2DTranspose(8, kernel_size=(2, 2)))
	model.add(BatchNormalization())
	model.add(Activation("relu"))

	model.add(Flatten())
	model.add(Dense(256))
	model.add(BatchNormalization())
	model.add(Dropout(.2))
	model.add(Activation("relu"))

	model.add(Dense(constants.INPUT_SEQUENCE_LENGTH * constants.NEURAL_INPUT_SIZE))
	model.add(Dropout(.2))
	model.add(BatchNormalization())

	model.add(Reshape((constants.INPUT_SEQUENCE_LENGTH, constants.NEURAL_INPUT_SIZE)))
	model.add(Activation(activation))
	
	model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
	return model


def create_CUDA_model():
	model = Sequential()

	model.add(CuDNNLSTM(layer_size, input_shape=input_shape, return_sequences=True))

	model.add(Bidirectional(CuDNNLSTM(256, return_sequences=True)))
	model.add(BatchNormalization())
	model.add(Dropout(0.2))

	model.add(Bidirectional(CuDNNLSTM(256, return_sequences=True)))
	model.add(BatchNormalization())
	model.add(Dropout(0.2))

	model.add(Bidirectional(CuDNNLSTM(256, return_sequences=True)))
	model.add(BatchNormalization())
	model.add(Dropout(0.2))

	model.add(TimeDistributed(Dense(output_shape)))
	model.add(Activation(activation))

	model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
	return model


def create_model():
	model = Sequential()

	model.add((LSTM(layer_size, input_shape=input_shape,
					activation='tanh', return_sequences=True)))
	model.add(BatchNormalization())
	model.add(Dropout(0.2))

	model.add(Bidirectional(LSTM(layer_size, activation='tanh')))
	model.add(RepeatVector(sequence_length))

	model.add(Bidirectional(
		LSTM(layer_size, activation='tanh', return_sequences=True)))
	model.add(BatchNormalization())
	model.add(Dropout(0.2))

	model.add(Bidirectional(
		LSTM(layer_size, activation='tanh', return_sequences=True)))
	model.add(TimeDistributed(Dense(output_shape)))
	model.add(Activation(activation))

	model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
	return model
