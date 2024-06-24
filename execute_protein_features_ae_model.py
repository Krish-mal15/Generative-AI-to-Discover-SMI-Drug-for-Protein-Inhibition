from keras.models import load_model
import numpy as np
from sklearn.model_selection import train_test_split

# Load and prprocess data for testing purposes
voxel_grids_array = np.load("voxel_grids_dataset.npy")

# Split into training and validation
X_train, X_val = train_test_split(voxel_grids_array, test_size=0.2, random_state=42)

X_train = X_train[..., np.newaxis]
X_val = X_val[..., np.newaxis]

loaded_encoder_model = load_model('voxel_encoder.keras')
encoded_features = loaded_encoder_model.predict(X_train[0])

print(encoded_features)

