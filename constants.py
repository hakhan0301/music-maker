

DATA_OUTPUT_LOCATION = "./Data/Processed/"
MIDI_INPUT_LOCATION = "./Data/PreProcessed/500 Batch - 1/"
MIDI_OUTPUT_LOCATION = "./Midi Output/"
MODEL_LOCATION = "./Model/Saved/"
PICKLE_FILE_NAME = "500 Batch - 1.obj"

TIME_OF_TIME_SLICE = .2
MAX_NOTE = 81
MIN_NOTE = 33
NOTES_COUNT = MAX_NOTE - MIN_NOTE + 1
NEURAL_INPUT_SIZE = NOTES_COUNT

INPUT_SEQUENCE_LENGTH = 20
# OUTPUT_SEQUENCE_LENGTH = 50

VALIDATION_DATA_SPLIT = .05
BATCHES = 64
EPOCHS = 30
LEARNING_RATE = 0.0007
DECAY_RATE = 0.0000007
TRAINING_LOSS = 'mean_squared_error'
TRAINING_METRICS = ['accuracy']