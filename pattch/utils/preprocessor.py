"""Preprocessing of molecules before feature calculation. This includes for example the identification of the aromatic C-H sites and the construction of respective RDKit mol objects."""

import logging
from typing import Tuple, Literal, Union
from rdkit import Chem


def get_mol_object(smiles: str) -> Union[Tuple[Literal[None], Literal[None]], Tuple[Chem.rdchem.Mol, str]]:
    """A Function for generating an RDKit mol object from a SMILES string.

    Parameters
    ----------
    smiles : str
        SMILES string of the molecule
    
    Returns
    -------
    mol : rdkit.Chem.rdchem.Mol or None
        RDKit mol object generated from the provided SMILES string. None if the mol object generation failed.
    canon_smiles : str
        Canonical SMILES string, generated from the mol object. None if the mol object generation failed.

    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        canon_smiles = Chem.MolToSmiles(mol)
        mol = Chem.MolFromSmiles(smiles)
        mol = Chem.AddHs(mol)
    except Exception:
        mol = None
        canon_smiles = None
    return mol, canon_smiles


class Preprocessor:
    """A class that implements the preprocessing pipeline of the PATTCH workflow.

    Parameters
    ----------
    info_dict : dict
        Dictionary with general information on the molecule (e.g., mol object, SMILES, potential sites of substitution).
    
    Attributes
    ----------
    substrate_mol : rdkit.Chem.rdchem.Mol
        RDKit mol object of the substrate molecule.
    substitution_sites : list
        Atom indices of all carbon atoms of aromatic C-H groups.
    unique_substitution_sites : dict
        Dictionary of all carbon atoms of aromatic C-H groups after removing symmetry-equivalent positions.
    adduct_mol : rdkit.Chem.rdchem.Mol
        RDKit mol object of the adduct that is formed with the substrate (e.g., with H+, Me+, or tBu+).
    adduct_smiles : str
        SMILES string of the adduct that is formed with the substrate (e.g., with H+, Me+, or tBu+).
    adduct_constraints : dict
        Dictionary for constraining internal coordinates (e.g., bond distances) of the adduct if required. This remains empty within the PATTCH workflow.
    
    Returns
    -------
    None
    
    """

    def __init__(self, info_dict: dict) -> None:
        # General attributes
        self.substrate_mol = info_dict["substrate_info"]["mol"]

        self.substitution_sites = []
        self.unique_substitution_sites = {}

        self.adduct_mol = None
        self.adduct_smiles = None
        self.adduct_constraints = {}

    def get_substitution_sites(self) -> None:
        """Method that finds all C-H bonds that are part of a aromatic system."""
        # Loop over all atoms of the molecule
        for atom in self.substrate_mol.GetAtoms():
            # Identify the aromatic C-H groups
            if atom.GetIsAromatic() is True and atom.GetSymbol() == "C":
                neighbor_symbols = [atom.GetSymbol() for atom in atom.GetNeighbors()]
                if neighbor_symbols.count("H") == 1:
                    self.substitution_sites.append(atom.GetIdx())

    def get_unique_substitution_sites(self) -> None:
        """Method that identifies which sites are symmetry equivalent and groups symmetry-equivalent sites."""
        sites = {}

        # Loop over the canonical rank atoms and save them to the sites dictionary
        for atom_idx, rank_idx in enumerate(list(Chem.CanonicalRankAtoms(self.substrate_mol, breakTies=False))):
            if rank_idx not in sites:
                sites[rank_idx] = [atom_idx]
            else:
                sites[rank_idx].append(atom_idx)

        # Formatting
        sites = {atom_indices[0]: atom_indices for atom_indices in sites.values()}

        # Define the unique substitution sites
        self.unique_substitution_sites = [atom_idx for atom_idx in self.substitution_sites if atom_idx in sites.keys()]

        # Formatting
        self.unique_substitution_sites = {pos: pos_list for pos, pos_list in sites.items() if pos in self.unique_substitution_sites}

    def build_adduct(self, substitution_atom_idx: int, adduct_type: str) -> None:
        """Method that builds a Wheland-type adduct at a specified site and for a selected adduct type."""
        self.adduct_mol = None
        self.adduct_smiles = None
        self.adduct_constraints = {}

        # Remove aromatic bond types from substrate to allow the formation of the Wheland complex, do that on a copy of the original substrate
        _mol = Chem.Mol(self.substrate_mol)
        Chem.Kekulize(_mol, clearAromaticFlags=True)

        # Different adducts can be constructed (H+, CH3+, tBu+, and the actual Wheland intermediate with the thianthrenium dication). In the final PATTCH workflow, the tBu+ adduct is used.
        if adduct_type == "proton":
            binding_mol = Chem.MolFromSmiles("[H+]")
            binding_atom_idx = len(list(_mol.GetAtoms()))
        elif adduct_type == "Me":
            binding_mol = Chem.MolFromSmiles("[H][C+]([H])[H]")
            binding_atom_idx = len(list(_mol.GetAtoms()))
        elif adduct_type == "tBu":
            binding_mol = Chem.MolFromSmiles("C[C+](C)C")
            binding_atom_idx = len(list(_mol.GetAtoms())) + 1
        else:
            binding_mol = Chem.MolFromSmiles("S1c2ccccc2Sc3ccccc13")
            binding_atom_idx = len(list(_mol.GetAtoms()))
        binding_mol = Chem.AddHs(binding_mol)

        # Add new bond between the substrate and the binding agent (H+, CH3+, etc.)
        self.adduct_mol = Chem.CombineMols(_mol, binding_mol)
        self.adduct_mol = Chem.RWMol(self.adduct_mol)
        self.adduct_mol.AddBond(binding_atom_idx, substitution_atom_idx, order=Chem.BondType.SINGLE)

        # Adjust charge of the binding agent's binding atom
        if adduct_type in ["proton", "Me", "tBu"]:
            self.adduct_mol.GetAtomWithIdx(binding_atom_idx).SetFormalCharge(0)
        else:
            self.adduct_mol.GetAtomWithIdx(binding_atom_idx).SetFormalCharge(1)

        # Adjust bond orders
        for bond in self.adduct_mol.GetAtomWithIdx(substitution_atom_idx).GetBonds():
            if bond.GetBondTypeAsDouble() == 2.0:
                bond.SetBondType(Chem.BondType.SINGLE)

                # Avoid setting the positive charge in the ring to where the substitution is going to happen
                ring_cationic_atom = bond.GetEndAtom()
                if ring_cationic_atom.GetIdx() == substitution_atom_idx:
                    ring_cationic_atom = bond.GetBeginAtom()

                real_n_radical_electrons = ring_cationic_atom.GetNumRadicalElectrons()
                ring_cationic_atom.SetNoImplicit(True)
                ring_cationic_atom.SetFormalCharge(ring_cationic_atom.GetFormalCharge() + 1)
                
                # Mark this atom for later for resetting the number of radical electrons
                ring_cationic_atom.SetAtomMapNum(42)
                break
        
        # Modify the adduct_object to make it to the transition state if requested
        if adduct_type == "TS":
            self._add_base(substitution_atom_idx)

        # Sanitize the adduct mol object
        Chem.SanitizeMol(self.adduct_mol)
        self.adduct_mol = Chem.Mol(self.adduct_mol)

        # Reset the number of unpaired electrons to the correct value. RDKit sometimes turns lone-pairs into two unpaired electrons which is not desired.
        for atom in self.adduct_mol.GetAtoms():
            if atom.GetAtomMapNum() == 42:
                atom.SetNumRadicalElectrons(real_n_radical_electrons)
                atom.SetAtomMapNum(0)

        # Get SMILES string of the adduct
        _mol = Chem.Mol(self.adduct_mol)
        _mol = Chem.RemoveHs(_mol)
        self.adduct_smiles = Chem.MolToSmiles(_mol)

        # Add bond distance constraints for the transition state
        if adduct_type == "TS":
            self._add_ts_constraints()

        logging.info("Adduct-%s: RDKit mol object generated.", substitution_atom_idx)
    
    def _add_base(self, substitution_atom_idx: int) -> None:
        """Method that modifies the adduct mol and turns it into the transition structure by adding a base for deprotonation."""
        # Set up base (diethyl ether), the O atom is the second in the mol object
        base_mol = Chem.MolFromSmiles("CCOCC")
        base_mol = Chem.AddHs(base_mol)

        # Find the proton that is going to be removed by the base
        h_atom_idx = [atom.GetIdx() for atom in self.adduct_mol.GetAtomWithIdx(substitution_atom_idx).GetNeighbors() if atom.GetSymbol() == "H"][0]

        # Add base to adduct mol object and form a bond between the base and the proton
        o_atom_idx = self.adduct_mol.GetNumAtoms() + 2
        self.adduct_mol = Chem.CombineMols(self.adduct_mol, base_mol)
        self.adduct_mol = Chem.RWMol(self.adduct_mol)
        self.adduct_mol.AddBond(h_atom_idx, o_atom_idx, order=Chem.BondType.SINGLE)

        # Add atomic charges
        self.adduct_mol.GetAtomWithIdx(h_atom_idx).SetFormalCharge(-1)
        self.adduct_mol.GetAtomWithIdx(o_atom_idx).SetFormalCharge(1)

        # Clean up the mol object
        self.adduct_mol = Chem.MolFromSmiles(Chem.MolToSmiles(self.adduct_mol))
        self.adduct_mol = Chem.AddHs(self.adduct_mol)

    def _add_ts_constraints(self) -> None:
        """Method for setting up the constraints dictionary for the constraint structure optimization of the transition state."""
        # Substructure for identifying the atom indices of the atoms that should be constraint
        substruc = Chem.MolFromSmarts("[S+X3]-[CH1X4]-[H-]-[O+X3](-[CH2]-[CH3])-[CH2]-[CH3]")

        # Identify atom indices
        match_indices = self.adduct_mol.GetSubstructMatch(substruc)
        s_idx = match_indices[0]
        c_idx = match_indices[1]
        h_idx = match_indices[2]
        o_idx = match_indices[3]

        # Define constraints dictionary, bond distances extracted from DOI: 10.1021/jacs.1c06281
        self.adduct_constraints = {"distances": [
            # S-C bond
            (s_idx, c_idx, 1.838),

            # C-H bond
            (c_idx, h_idx, 1.25),

            # H-O bond
            (h_idx, o_idx, 1.458)
        ]}
