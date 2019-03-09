import constants
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, RepeatVector, BatchNormalization, Flatten, TimeDistributed


input_shape = [constants.INPUT_SEQUENCE_LENGTH, constants.NOTES_COUNT]
output_shape =  constants.NOTES_COUNT
sequence_length = constants.INPUT_SEQUENCE_LENGTH

def create_model():
	model = Sequential()
	
	model.add(LSTM(units = 96, input_shape = input_shape, activation = 'tanh', return_sequences = True))
	model.add(BatchNormalization())
	model.add(Dropout(0.2))

	model.add(LSTM(96, activation = 'tanh'))
	model.add(RepeatVector(sequence_length))

	model.add(LSTM(96, activation = 'tanh', return_sequences = True))
	model.add(BatchNormalization())
	model.add(Dropout(0.2))

	model.add(LSTM(96, activation = 'tanh', return_sequences = False))
	model.add(Dropout(0.2))
	model.add(BatchNormalization())
	# model.add(TimeDistributed(Dense(48, activation = 'softmax')))
	model.add(Dense(output_shape, activation='softmax'))

	return model