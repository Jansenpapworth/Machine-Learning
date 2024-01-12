from tensorflow.keras.models import load_model

# Load the model from the H5 file
model = load_model('jp1120-1.h5')

# Display the summary of the model
model.summary()
