# 1. Get PDB File
# 2. Turn PDB File into Voxels
# 2. CNN Encoder Model Predicts Feature Vector
# 3. Extract Sequence from PDB File
# 4. Get Protein Descriptors
# 5. Reshape Protein Descriptors
# 6. Concatenate Feature Vectors and Descriptor

from Bio.PDB.PDBParser import PDBConstructionWarning
import numpy as np
import os

from preprocess_protein_structure import parse_pdb_and_get_coordinates, create_voxel_grid
from preprocess_protein_descriptors import ProteinDescriptorDataset

from keras.models import load_model
from Bio import PDB
from Bio.SeqUtils import seq1
import warnings

import tensorflow as tf

warnings.simplefilter("ignore", PDBConstructionWarning)

voxel_size = 1.0
grid_resolution = 32

main_directory = 'v2019-other-PL'
main_directory_test = 'main_directory_test'


# voxel_model = load_model('voxel_encoder.keras')


def extract_sequence_from_pdb(pdb_file):
    parser = PDB.PDBParser()
    structure = parser.get_structure('structure', pdb_file)

    # Get the first model in the structure
    model = structure[0]
    sequence = ''

    # Iterate over all chains in the model
    for chain in model:
        for residue in chain:
            if PDB.is_aa(residue, standard=True):
                sequence += seq1(residue.resname)

    return sequence


def create_protein_main_dataset(main_directory):
    protein_main_dataset_array = []

    # Walk through main directory and its subdirectories
    for root, dirs, files in os.walk(main_directory):
        for file in files:
            if file.endswith("_protein.pdb"):  # Check if file is a PDB file
                file_path = os.path.join(root, file)
                print("Processing: ", file)

                protein_aa_sequence = extract_sequence_from_pdb(file_path)
                protein_encoded_descriptors = ProteinDescriptorDataset(
                    protein_sequence_list_data=[protein_aa_sequence]).read()
                reshaped_protein_descriptor_array = np.tile(protein_encoded_descriptors, (32, 4, 4, 1, 1))
                # print(reshaped_protein_descriptor_array.shape)

                atom_coordinates = parse_pdb_and_get_coordinates(file_path)
                voxel_grid = create_voxel_grid(atom_coordinates)
                voxel_grid = voxel_grid[..., np.newaxis]
                feature_vector = voxel_model.predict(voxel_grid)
                # print(feature_vector)

                # This is the array with the structural feature vector and the
                # protein chemical descriptor concatenated into one array
                protein_data_array = np.concatenate((feature_vector, reshaped_protein_descriptor_array), axis=-1)
                protein_main_dataset_array.append(protein_data_array)

    np.save("main_protein_dataset.npy", protein_main_dataset_array)


# create_protein_main_dataset(main_directory)
# print(np.load('main_protein_dataset.npy'))

def protein_embedding(protein_data_array, embedding_dims):
    print(protein_data_array.shape)
    B, T, H, W, D, C = protein_data_array.shape

    # Ensure the vocab size is appropriate
    vocab_size = int(np.max(protein_data_array) + 1)
    print(f"Vocab size: {vocab_size}")

    embedding = tf.keras.Sequential([
        tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embedding_dims, input_shape=(C,))
    ])

    # Reshape the input data to a 2D shape (flatten the spatial dimensions)
    flat_data = protein_data_array.reshape(-1, C)

    # Embed the protein data
    embedded_protein = embedding.predict(flat_data)

    # Reshape the embedded tensor to the original 6D shape with embedding dimensions
    embedded_protein = embedded_protein.reshape(B, T, H, W, D, C, embedding_dims)

    return embedded_protein


dataset_testing = np.load('main_protein_dataset.npy')
embedding_dims = 128
protein_embedded = protein_embedding(dataset_testing, embedding_dims)
print(protein_embedded.shape)
