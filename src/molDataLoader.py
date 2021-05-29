"""Convert smile string into some useful features."""
import torch
import itertools
import pickle
from rdkit import Chem
"""DATA PREPROCESSING - BUILD A DATASET WITH MOLECULE FEATURES"""
# data_dir = '/home/gonzo/Downloads/data.pkl'
# with open(data_dir, 'rb') as f:
#     data = pickle.load(f)
# data = dict(itertools.islice(data.items(), 0, 20))
data_dir = '/home/gonzo/Downloads/chosen_dict_energy_smile.pt'
data = torch.load(data_dir)
# Get energies and molecules from SMILES string.
# https://www.rdkit.org/docs/source/rdkit.Chem.rdchem.html
energies = []
molecules = []
for molecule in data:
    energies.append(data[molecule][0])
    molecules.append(Chem.rdmolfiles.MolFromSmiles(data[molecule][1]))
print(dir(molecules[0]))
# Molecule-level features.
m = molecules[0]
print(Chem.MolToSmiles(m))
m.GetNumAtoms()       # Number of atoms in the conformer
m.GetNumBonds()       # Number of bonds in the molecule
m.GetNumConformers()  # Number of conformations on the molecule. All are zero, why?
m.GetNumHeavyAtoms()  # Number of heavy atoms (atomic number >1) in the molecule
m.GetPropNames()      # Tuple with all property names for this molecule
# Atom-level features.
# To do: Functional groups (how many carbons, oxygen, etc).
atom = next(m.GetAtoms())
print(dir(atom))
for atom, bond in zip(m.GetAtoms(), m.GetBonds()):
    print(f'atomic num: {atom.GetAtomicNum()}, '
          f'bond type: {bond.GetBondType()}, '
          f'aromatic atom: {atom.GetIsAromatic()}')
dataset = torch.tensor([])
for molecule, energy in zip(molecules, energies):
    num_aromatic = 0
    for atom in molecule.GetAromaticAtoms():
        if atom.GetIsAromatic():
            num_aromatic += 1
    dataset = torch.cat((dataset, torch.tensor([energy, molecule.GetNumAtoms(),
                                                molecule.GetNumHeavyAtoms(),
                                                molecule.GetNumBonds(),
                                                num_aromatic]).unsqueeze(0)), dim=0)