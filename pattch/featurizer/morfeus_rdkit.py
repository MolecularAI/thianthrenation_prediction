"""Feature caluclation with Morfeus and RDKit."""

import os
import logging
import numpy as np
from typing import Tuple, Callable
from rdkit import Chem
from morfeus import Sterimol, BuriedVolume, SASA, read_xyz


class MorfeusRDKit:
    """A Class for calculating all features from morfeus and RDKit for the PATTCH workflow.

    Parameters
    ----------
    compound_dir : str
        Path to the folder of the currently treated molecule.
    info_dict : dict
        Dictionary with general information on the molecule (e.g., mol object, SMILES, potential sites of substitution).
    feature_dict : dict
        Feature dictionary with the atom indices as keys. The values are dictionaries with the feature name as keys and the numbers as values.
    mol_name : str
        Name of the currently treated compound as specified by the user.
    atom_indices : list
        List of atom indices for which features should be calculated.

    Attributes
    ----------
    substrate_mol : rdkit.Chem.rdchem.Mol
        RDKit mol object of the substrate molecule.
    substrate_elements : list
        Atom types as specified by atomic symbol of the substrate molecule.
    substrate_coordinates : list
        xyz coordinates of the substrate molecule.
    substrate_rdkit_ring_info : rdkit.Chem.rdchem.RingInfo
        Information on the rings of the molecule.
    substrate_rdkit_bond_distance_matrix : numpy.ndarray
        Bond distance matrix as calculated with RDKit for the substrate molecule.
    root_atom_idx : int
        Atom index of the carbon atom of the currently treated C-H group (site-specific attribute).
    root_atom_ring_atom_indices : list
        Atom indices of the atoms of the ring the currently treated C-H group is part of (site-specific attribute).
    ortho_atom_indices : list
        Atom indices of the atoms ortho to the currently treated C-H group (site-specific attribute).
    meta_atom_indices : list
        Atom indices of the atoms meta to the currently treated C-H group (site-specific attribute).
    para_atom_indices : list
        Atom indices of the atoms para to the currently treated C-H group (site-specific attribute).
    sterimol_axes : list
        Atom indice of the Sterimol axis of the two ortho substituents of the currently treated CH group (site-specific attribute).
    sterimol_data : numpy.ndarray
        Sterimol parameters of the two ortho substituents of the currently treated CH group(site-specific attribute).
    
    Returns
    -------
    feature_dict : dict
        Feature dictionary updated with the new features.
    
    """

    def __init__(
        self,
        compound_dir: str,
        info_dict: dict,
        feature_dict: dict,
        mol_name: str,
        atom_indices: list,
    ) -> None:
        self.compound_dir = compound_dir
        self.info_dict = info_dict
        self.feature_dict = feature_dict
        self.mol_name = mol_name
        self.atom_indices = atom_indices

        # General attributes
        self.substrate_mol = None
        self.substrate_elements = None
        self.substrate_coordinates = None
        self.substrate_rdkit_ring_info = None
        self.substrate_rdkit_bond_distance_matrix = None

        # CH-group-specific attributes
        self.root_atom_idx = None
        self.root_atom_ring_atom_indices = None
        self.ortho_atom_indices = None
        self.meta_atom_indices = None
        self.para_atom_indices = None
        self.sterimol_axes = None
        self.sterimol_data = None

    def get_values(self) -> dict:
        """Method that executes the feature generation workflow."""
        # Change to the folder of the compound
        os.chdir(self.compound_dir)

        # Run workflow for all relevant atoms
        self._set_general_info()
        for atom_idx in self.atom_indices:
            self._run_workflow(atom_idx)

        # Change back to the main directory and return updated feature dictionary
        os.chdir("../..")
        return self.feature_dict

    def _set_general_info(self) -> None:
        """Method for setting certain substrate-specific class attributes relevant for each position."""
        self.substrate_mol = self.info_dict["substrate_info"]["mol"]
        self.substrate_rdkit_ring_info = self.substrate_mol.GetRingInfo()
        self.substrate_rdkit_bond_distance_matrix = Chem.GetDistanceMatrix(self.substrate_mol)

        self.substrate_elements, self.substrate_coordinates = self._read_xyz_file(f"XTBREFINED_{self.mol_name}.xyz")

    def _run_workflow(self, root_atom_idx: int) -> None:
        """Method for calculating the features for a given carbon atom index of an aromatic CH group."""
        # CH-group-specific attributes
        self.root_atom_idx = root_atom_idx
        self.root_atom_ring_atom_indices = []
        self.ortho_atom_indices = []
        self.meta_atom_indices = []
        self.para_atom_indices = []
        self.sterimol_axes = []
        self.sterimol_data = None

        # Individual features
        primary_features = {
            # Morfeus
            "morfeus_ortho_sterimol_L": self._get_morfeus_ortho_sterimol_l,
            "morfeus_ortho_sterimol_B1": self._get_morfeus_ortho_sterimol_b1,
            "morfeus_ortho_sterimol_B5": self._get_morfeus_ortho_sterimol_b5,
            "morfeus_buried_volume": self._get_morfeus_buried_volume,
            "morfeus_ortho_buried_volume": self._get_morfeus_ortho_buried_volume,
            "morfeus_sasa": self._get_morfeus_sasa,
            "morfeus_ortho_sasa": self._get_morfeus_ortho_sasa,
            # RDKit
            "rdkit_neighbor_count": self._get_rdkit_neighbor_count,
            "rdkit_ring_size": self._get_rdkit_ring_size,
            "rdkit_hetero_cycle": self._get_rdkit_hetero_cycle,
            "rdkit_alpha_to_hetero_atom": self._get_rdkit_alpha_to_hetero_atom,
        }

        # Run setup methods
        self._setup()

        # Calculate features and store them in the feature dictionary
        for feature_name, method in primary_features.items():
            self._set_feature(feature_name, method)

    def _setup(self) -> None:
        """Method for executing data calculation prior to the actual feature calculation."""
        self._setup_ortho_meta_para_assignment()
        self._setup_ortho_sterimol_axes()
        self._setup_morfeus_ortho_sterimol_data()

    def _setup_ortho_meta_para_assignment(self) -> None:
        """Method for identifying ortho, meta, and para atom indices."""
        # Identify the ring system of the currently treated CH group
        for ring_system in self.substrate_rdkit_ring_info.AtomRings():
            if self.root_atom_idx in ring_system:
                self.root_atom_ring_atom_indices = ring_system

        # Get the distances of the individual atoms of the identified ring system to the currently treated CH group (measured in number of bonds)
        ring_bond_distances = [int(self.substrate_rdkit_bond_distance_matrix[self.root_atom_idx, atom_idx]) for atom_idx in self.root_atom_ring_atom_indices]

        # Assign the atoms either as ortho, meta, or para
        for atom_idx, bond_distance in zip(self.root_atom_ring_atom_indices, ring_bond_distances):
            if bond_distance == 1:
                self.ortho_atom_indices.append(atom_idx)
            if bond_distance == 2:
                self.meta_atom_indices.append(atom_idx)
            if bond_distance == 3:
                self.para_atom_indices.append(atom_idx)

        # Store ortho/meta/para assignment
        for effective_atom_idx, sub_data in self.info_dict["site_info"].items():
            if self.root_atom_idx in sub_data["site_indices"]:
                break

        self.info_dict["site_info"][effective_atom_idx]["ortho_atom_indices"][str(self.root_atom_idx)] = self.ortho_atom_indices
        self.info_dict["site_info"][effective_atom_idx]["meta_atom_indices"][str(self.root_atom_idx)] = self.meta_atom_indices
        self.info_dict["site_info"][effective_atom_idx]["para_atom_indices"][str(self.root_atom_idx)] = self.para_atom_indices

    def _setup_ortho_sterimol_axes(self) -> None:
        """Method for obtaining the Sterimol axis of the substituents ortho to the currently treated CH group."""
        # Loop over the ortho atom indices
        for first_atom_idx in self.ortho_atom_indices:
            ortho_atom = self.substrate_mol.GetAtomWithIdx(first_atom_idx)
            first_atom_neighbor_indices = [atom.GetIdx() for atom in ortho_atom.GetNeighbors()]

            # Find the second atom for defining the Sterimol axis
            second_atom_indices = []
            for atom_idx in first_atom_neighbor_indices:
                if (atom_idx != self.root_atom_idx and atom_idx not in self.root_atom_ring_atom_indices):
                    second_atom_indices.append(atom_idx)

            # Store the found sterimol axis
            if (len(second_atom_indices) < 1):  # in case of a neighboring sp2 nitrogen or sulfur
                self.sterimol_axes.append((first_atom_idx, None))
            else:
                self.sterimol_axes.append((first_atom_idx, second_atom_indices[0]))

    def _setup_morfeus_ortho_sterimol_data(self) -> None:
        """Method for calculating the Sterimol parameters."""
        # List for results
        sterimol_parameters = []

        # Loop over the sterimol axis
        for ax in self.sterimol_axes:
            # Zero-padding in case there is no substituent in ortho position
            if None in ax:
                sterimol_parameters.append([0.0, 0.0, 0.0])
            else:
                # Determine which atoms should be used for calculating the Sterimol parameters
                exclude_atom_indices = self._get_sterimol_atom_exclusion_indices(ax)

                # Calculate the values and append them to the results list
                sterimol = Sterimol(
                    self.substrate_elements,
                    self.substrate_coordinates,
                    ax[0] + 1,
                    ax[1] + 1,
                    excluded_atoms=[idx + 1 for idx in exclude_atom_indices],
                )
                sterimol.bury(method="truncate")
                sterimol_parameters.append([sterimol.L_value, sterimol.B_1_value, sterimol.B_5_value])

        # Calculate the sum of the Sterimol parameters of the two ortho sites
        self.sterimol_data = np.sum(sterimol_parameters, axis=0)

    def _get_sterimol_atom_exclusion_indices(self, sterimol_ax: tuple) -> list:
        """
        Method that returns all atom indices that do not belong to the substituent for which the Sterimol parameters should be calculated
        1) Identify the ring atom indices of the ring of the current root atom
        2) Delete all of these ring atoms except for the atom ortho to the root atom --> molecule falls apart
        3) Identify the fragment that contains the ortho atom
        4) Return all non-ortho substituent atom indices
        """
        ortho_atom_idx = sterimol_ax[0]

        # Define new editable RDKit mol object
        rw_mol = Chem.RWMol(self.substrate_mol)

        # Set atom mapping numbers to atom indices for keeping track of atoms
        for atom in rw_mol.GetAtoms():
            atom.SetAtomMapNum(atom.GetIdx())

        # Remove ring atoms from the mol object if they are not the ortho atom --> molecule gets fragmented
        root_atom_ring = [idx for idx in self.root_atom_ring_atom_indices]
        root_atom_ring.sort()
        for atom_idx in root_atom_ring[::-1]:
            if atom_idx != ortho_atom_idx:
                rw_mol.RemoveAtom(atom_idx)

        # Loop over the fragments and identify those which contain the ortho atom
        for frag in Chem.GetMolFrags(rw_mol):
            old_atom_indices = []
            for idx in frag:
                old_atom_indices.append(rw_mol.GetAtomWithIdx(idx).GetAtomMapNum())

            # Return the atom indices that should be excluded when calculating the Sterimol parameters
            if ortho_atom_idx in old_atom_indices:
                return [idx for idx in range(len(self.substrate_mol.GetAtoms()))if idx not in old_atom_indices]

    def _set_feature(self, feature_name: str, get_method: Callable[[], float]) -> None:
        """Method for writing the feature values to the feature dictionary."""
        try:
            self.feature_dict[str(self.root_atom_idx)][feature_name] = get_method()
        except Exception as e:
            logging.error("Adduct-%s: in MorfeusRDKit._set_feature(): %s could not be successfully calculated: %s: %s", self.root_atom_idx, feature_name, e.__class__.__name__, e)

    def _get_morfeus_ortho_sterimol_l(self) -> float:
        """Method for getting the Sterimol L value."""
        return float(self.sterimol_data[0])

    def _get_morfeus_ortho_sterimol_b1(self) -> float:
        """Method for getting the Sterimol B1 value."""
        return float(self.sterimol_data[1])

    def _get_morfeus_ortho_sterimol_b5(self) -> float:
        """Method for getting the Sterimol B5 value."""
        return float(self.sterimol_data[2])

    def _get_morfeus_buried_volume(self) -> float:
        """Method for getting the buried volume value."""
        bv = BuriedVolume(
            self.substrate_elements,
            self.substrate_coordinates,
            self.root_atom_idx + 1,
            include_hs=True,
            radius=2.8,
        )
        return float(bv.fraction_buried_volume)

    def _get_morfeus_ortho_buried_volume(self) -> float:
        """Method for getting the buried volume value of the ortho-atoms as a mean value."""
        buried_volumes = []
        for atom_idx in self.ortho_atom_indices:
            bv = BuriedVolume(
                self.substrate_elements,
                self.substrate_coordinates,
                atom_idx + 1,
                include_hs=True,
                radius=2.8,
            )
            buried_volumes.append(bv.fraction_buried_volume)
        return float(np.sum(buried_volumes) / 2)
    
    def _get_morfeus_sasa(self):
        """Method for getting the solvent accessible surface area value."""
        sasa = SASA(self.substrate_elements, self.substrate_coordinates)
        return sasa.atom_areas[self.root_atom_idx+1]
    
    def _get_morfeus_ortho_sasa(self):
        """Method for getting the solvent accessible surface area of the ortho-atoms as a mean value."""
        sasa_s = []
        for atom_idx in self.ortho_atom_indices:
            sasa = SASA(self.substrate_elements, self.substrate_coordinates)
            sasa_s.append(sasa.atom_areas[atom_idx+1])
        return float(np.sum(sasa_s) / 2)

    def _get_rdkit_neighbor_count(self) -> int:
        """Method for getting the number of neighbors of a given CH group. The neighbor count is increased by 1 if the neighboring group is not a CH but instead a C-R group."""
        # Neighbor count
        neighbor_count = 0

        # Loop over the sterimol axis
        for axis in self.sterimol_axes:
            if None in axis:
                continue

            # Check which atom types define the sterimol axis
            symbols = []
            for idx in axis:
                symbols.append(self.substrate_mol.GetAtomWithIdx(idx).GetSymbol())
            symbols.sort()
            if symbols != ["C", "H"]:
                neighbor_count += 1
        
        return neighbor_count

    def _get_rdkit_ring_size(self) -> float:
        """Method for getting the ring size the currently treated CH group is part of."""
        return sorted(self.substrate_rdkit_ring_info.AtomRingSizes(self.root_atom_idx))[0]

    def _get_rdkit_hetero_cycle(self) -> int:
        """Method for determining of the currently treated CH group is part of a heterocycle."""
        ring_system_idx = sorted(self.substrate_rdkit_ring_info.AtomMembers(self.root_atom_idx))[0]
        ring_system = self.substrate_rdkit_ring_info.AtomRings()[ring_system_idx]
        ring_system_symbols = set([self.substrate_mol.GetAtomWithIdx(idx).GetSymbol() for idx in ring_system])
        if len(ring_system_symbols) == 1:
            return 0
        return 1

    def _get_rdkit_alpha_to_hetero_atom(self) -> int:
        """Method for checking if the currently treated CH group is in direct vicinity of a hetero atom."""
        hetero_atoms = ["O", "S", "N"]
        root_atom_neighbors = self.substrate_mol.GetAtomWithIdx(self.root_atom_idx).GetNeighbors()
        for atom in root_atom_neighbors:
            if atom.GetSymbol() in hetero_atoms:
                return 1
        return 0

    @staticmethod
    def _read_xyz_file(file_path: str) -> Tuple[list, list]:
        """Static method for reading an xyz file in XMOL format with morfeus."""
        elements, coordinates = read_xyz(file_path)
        return elements, coordinates
