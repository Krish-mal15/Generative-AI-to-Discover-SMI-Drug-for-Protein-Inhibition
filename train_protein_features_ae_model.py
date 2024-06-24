from protein_structure_cnn_model import build_simple_cnn, build_autoencoder
from sklearn.model_selection import train_test_split
import numpy as np
from keras.optimizers import Adam
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint
from sklearn.decomposition import PCA
import umap


# Load Dataset
voxel_grids_array = np.load("voxel_grids_dataset.npy")

# Split into training and validation
X_train, X_val = train_test_split(voxel_grids_array, test_size=0.2, random_state=42)
print(f"Training set shape: {X_train.shape}")
print(f"Validation set shape: {X_val.shape}")

# Instantiate and compile model
input_shape = (32, 32, 32, 1)
autoencoder, encoder = build_autoencoder(input_shape)
autoencoder.summary()
encoder.summary()
autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

# Add dimensionality to voxel dataset
X_train = X_train[..., np.newaxis]
X_val = X_val[..., np.newaxis]

callback = ModelCheckpoint(filepath='best_autoencoder_protein_cnn.keras',
                           monitor='val_loss',
                           mode='min',
                           save_best_only=True,
                           verbose=1)

# Train model
history = autoencoder.fit(X_train, X_train,
                          epochs=5,
                          batch_size=16,
                          shuffle=True,
                          validation_data=(X_val, X_val),
                          callbacks=[callback])

# Save model
autoencoder.save('voxel_autoencoder.keras')
encoder.save('voxel_encoder.keras')


def plot_training_history(history):
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


plot_training_history(history)

# Test on the training data to see if it works
encoded_features = encoder.predict(X_train)
print(f"Shape of feature vectors: {encoded_features.shape}")

flat_encoded_features = encoded_features.reshape(encoded_features.shape[0], -1)

# Apply PCA to reduce the dimensions to 2 for visualization
pca = PCA(n_components=2)
pca_result = pca.fit_transform(flat_encoded_features)

# Plot the PCA result
plt.figure(figsize=(8, 6))
plt.scatter(pca_result[:, 0], pca_result[:, 1], c='blue', alpha=0.5)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA of Encoded Feature Vectors')
plt.show()


