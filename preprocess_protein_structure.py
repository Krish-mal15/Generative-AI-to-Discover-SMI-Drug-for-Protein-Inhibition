from Bio.PDB import PDBParser
import numpy as np
from Bio import PDB
import os

main_directory = 'v2019-other-PL'
voxel_size = 1.0
grid_resolution = 32


# Function to parse PDB and extract coordinates
def parse_pdb_and_get_coordinates(file_path):
    parser = PDBParser()
    structure = parser.get_structure('protein', file_path)

    atom_coords = []
    for model in structure:
        for chain in model:
            for residue in chain:
                for atom in residue:
                    atom_coords.append(atom.get_coord())

    return atom_coords


# Function to create voxel grid from coordinates
def create_voxel_grid(atom_coordinates):
    voxel_grid = np.zeros((grid_resolution, grid_resolution, grid_resolution))

    def map_to_voxel_index(coord):
        # Round and clip to ensure indices are within bounds
        idx = tuple(int(np.clip(np.round(coord[i] / voxel_size), 0, grid_resolution - 1)) for i in range(3))
        return idx

    # Populate voxel grid
    for coord in atom_coordinates:
        idx = map_to_voxel_index(coord)
        voxel_grid[idx] = 1

    # Normalize voxel grid
    voxel_grid /= np.max(voxel_grid)

    return voxel_grid


# Function to recursively parse PDB files in directory and create voxel grids dataset
def create_voxel_grids_dataset(main_directory):
    voxel_grids_dataset = []

    # Walk through main directory and its subdirectories
    for root, dirs, files in os.walk(main_directory):
        for file in files:
            if file.endswith("_protein.pdb"):  # Check if file is a PDB file
                file_path = os.path.join(root, file)
                atom_coordinates = parse_pdb_and_get_coordinates(file_path)
                voxel_grid = create_voxel_grid(atom_coordinates)
                voxel_grids_dataset.append(voxel_grid)

                print(f"Processed: {file}")

                print(np.array(voxel_grids_dataset))

    return np.array(voxel_grids_dataset)

# Create voxel grids dataset
# voxel_grids_dataset = create_voxel_grids_dataset(main_directory)
# voxel_grids_array = np.array(voxel_grids_dataset)
# np.save("voxel_grids_dataset.npy", voxel_grids_array)
