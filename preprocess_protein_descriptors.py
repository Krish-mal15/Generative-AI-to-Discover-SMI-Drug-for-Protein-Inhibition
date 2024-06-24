import numpy as np
from PyBioMed.PyProtein import CTD
from PyBioMed.PyProtein import AAComposition as AAC
import pandas as pd

data = pd.read_csv('main_bind_dataset.csv')
data.drop(labels=['pdbid', 'COLUMN', 'pocket'], axis=1, inplace=True)


def encode_sequence(protein_aa_sequence):
    protein_descriptors_ctd = CTD.CalculateCTD(protein_aa_sequence)
    protein_properties_list_ctd = list(protein_descriptors_ctd.values())

    protein_descriptors_comp = AAC.CalculateAAComposition(protein_aa_sequence)
    protein_properties_list_comp = list(protein_descriptors_comp.values())

    protein_properties_list_main = np.concatenate((protein_properties_list_comp, protein_properties_list_ctd))

    return protein_properties_list_main


class ProteinDescriptorDataset:
    def __init__(self, protein_sequence_list_data):
        self.protein_sequence_list_data = protein_sequence_list_data

    def read(self):
        encoded_sequences = []

        for protein in self.protein_sequence_list_data:
            encoded_sequence = encode_sequence(protein)
            encoded_sequences.append(encoded_sequence)

        return np.array(encoded_sequences)


# Example:
# protein_encoded_descriptors = ProteinDescriptorDataset(protein_sequence_list_data=data['sequence'].to_list()).read()
# print(protein_encoded_descriptors)
