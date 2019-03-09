import constants
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, RepeatVector, TimeDistributed, BatchNormalization

INPUT_SHAPE = constants.NOTES_COUNT
OUTPUT_SHAPE =  constants.NOTES_COUNT

def create_model():
	model = Sequential()
	
	model.add(LSTM(units = 96, input_dim = INPUT_SHAPE, activation = 'tanh', return_sequences = True))
	model.add(BatchNormalization())
	model.add(Dropout(0.2))

	model.add(LSTM(96, activation = 'tanh'))
	model.add(BatchNormalization())
	model.add(RepeatVector(48))

	model.add(LSTM(96, activation = 'tanh', return_sequences = True))
	model.add(BatchNormalization())
	model.add(Dropout(0.2))

	model.add(LSTM(96, activation = 'tanh', return_sequences = True))
	model.add(BatchNormalization())
	model.add(Dropout(0.2))

	model.add(TimeDistributed(Dense(OUTPUT_SHAPE, activation = 'softmax')))

	return model