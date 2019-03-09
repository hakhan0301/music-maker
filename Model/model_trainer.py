LOSS = 'categorical_crossentropy'
# OPTIMIZER = tf.keras.optimizers.Adam(lr = 7e-4, decay = 1e-5)
METRICS = ['accuracy']

def train_model(model, training_data):
	model.compile(loss = LOSS, optimizer = OPTIMIZER, METRICS = METRICS)
	return