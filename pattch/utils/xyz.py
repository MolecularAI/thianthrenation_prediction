"""Check xzy structures after optimization."""

from rdkit import Chem
from rdkit.Chem import rdDetermineBonds


class CheckConnectivity:
    """A class to check if the atom connectivity determined from an xyz file matches that of a mol object generated from a SMILES string.

    Arguments
    ---------
    mol : rdkit.Chem.rdchem.Mol
        RDKit mol object of the molecule.
    xyz_file_path : str
        Path to the xyz file (must be in XMOL format).

    Returns
    -------
    bool
        True if the atom connectivity matches; False otherwise.
    """

    def __init__(self, mol: Chem.rdchem.Mol, xyz_file_path: str) -> None:
        self.mol = mol
        self.xyz_file_path = xyz_file_path

    def check(self) -> bool:
        """Method that compares two atom connectivity dictionaries."""
        bond_dict_xyz = self._get_bond_dict_from_xyz()
        bond_dict_mol = self._get_bond_dict(self.mol)

        if not (bond_dict_mol == bond_dict_xyz):
            return False
        return True

    def _get_bond_dict_from_xyz(self) -> dict:
        """Method that generates an atom connectivity dictionary from an xyz file."""
        raw_mol = Chem.MolFromXYZFile(self.xyz_file_path)
        mol = Chem.Mol(raw_mol)
        rdDetermineBonds.DetermineConnectivity(mol)
        return self._get_bond_dict(mol)

    @staticmethod
    def _get_bond_dict(mol) -> dict:
        """Static method that generates an atom connectivity dictionary from an RDKit mol object."""
        # Final results dictionary
        bond_dict = {}

        # Loop over all bonds of the molecule
        for bond in mol.GetBonds():
            idx_1 = bond.GetBeginAtom().GetIdx()
            idx_2 = bond.GetEndAtom().GetIdx()

            # Store for every atom its neighbor atom indices
            if idx_1 not in bond_dict:
                bond_dict[idx_1] = [idx_2]
            else:
                bond_dict[idx_1].append(idx_2)

            if idx_2 not in bond_dict:
                bond_dict[idx_2] = [idx_1]
            else:
                bond_dict[idx_2].append(idx_1)

        # Sort the neighbor lists
        for neighbor_list in bond_dict.values():
            neighbor_list.sort()

        # Sort the final atom connectivity dict
        bond_dict = dict(sorted(bond_dict.items(), reverse=False))
        return bond_dict
