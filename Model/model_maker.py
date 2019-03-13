import constants
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, RepeatVector, BatchNormalization, TimeDistributed, Bidirectional, Activation, Flatten, CuDNNLSTM

input_shape = [constants.INPUT_SEQUENCE_LENGTH, constants.NEURAL_INPUT_SIZE]
output_shape =  constants.NEURAL_INPUT_SIZE
sequence_length = constants.INPUT_SEQUENCE_LENGTH

lr = constants.LEARNING_RATE
decay = constants.DECAY_RATE
loss = constants.TRAINING_LOSS
metrics = constants.TRAINING_METRICS
optimizer = tf.keras.optimizers.Adam(lr = lr, decay = decay)
activation = constants.ACTIVATION


def create_nobi_model(): 
	model = Sequential()
	
	model.add((LSTM(96, input_shape=input_shape, activation = 'relu', return_sequences = True)))
	model.add(BatchNormalization())
	model.add(Dropout(0.2))

	model.add((LSTM(96, activation = 'relu')))
	model.add(RepeatVector(sequence_length))

	model.add((LSTM(96, activation = 'relu', return_sequences = True)))
	model.add(BatchNormalization())
	model.add(Dropout(0.2))

	model.add((LSTM(96, activation = 'relu', return_sequences = True)))
	model.add(TimeDistributed(Dense(output_shape)))
	model.add(Activation(activation))

	model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
	return model

def create_model(): 
	model = Sequential()
	
	model.add((LSTM(192, input_shape=input_shape, activation = 'relu', return_sequences = True)))
	model.add(BatchNormalization())
	model.add(Dropout(0.2))

	model.add(Bidirectional(LSTM(192, activation = 'relu')))
	model.add(RepeatVector(sequence_length))

	model.add(Bidirectional(LSTM(192, activation = 'relu', return_sequences = True)))
	model.add(BatchNormalization())
	model.add(Dropout(0.2))

	model.add(Bidirectional(LSTM(192, activation = 'relu', return_sequences = True)))
	model.add(TimeDistributed(Dense(output_shape)))
	model.add(Activation(activation))

	model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
	return model