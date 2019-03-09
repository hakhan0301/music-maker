from constants import NOTES_COUNT

INPUT_SHAPE = [NOTES_COUNT]
OUTPUT_SHAPE = [NOTES_COUNT]

def create_model():
	model = Sequential()

	model.add(LSTM(input_shape = INPUT_SHAPE, activation = 'tanh', return_sequences = True))
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

	return model()