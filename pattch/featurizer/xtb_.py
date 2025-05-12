"""Feature calculation with xtb through the XTB and the XTBChargeFukui class."""

import os
import shutil
import logging
from typing import Callable
from subprocess import PIPE, run

from ..featurizer.conformer_pipeline import MolPipeline


class XTB:
    """A class for the calculation of the xtb energy features (approximate transition state energy or H+/Me+/tBu+ affinities) of the PATTCH workflow.
    
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
    adduct_type : str
        Name of the chemical agent that is used to probe a given C-H group. "tBu" for PATTCH.
    
    Attributes
    ----------
    substrate_xtb_energy : float
        Electronic energy of the substrate molecule.
    root_atom_idx : int
        Atom index of the carbon atom of the currently treated C-H group.
    
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
            adduct_type: str,
            ) -> None:
        self.compound_dir = compound_dir
        self.info_dict = info_dict
        self.feature_dict = feature_dict
        self.mol_name = mol_name
        self.adduct_type = adduct_type

        # General attributes
        self.substrate_xtb_energy = None

        # CH-group-specific attributes
        self.root_atom_idx = None

    def get_values(self) -> dict:
        """Method for running the workflow."""
        # Change to the folder of the compound
        os.chdir(self.compound_dir)

        # Run workflow for the individual sites
        try:
            self.substrate_xtb_energy = self._read_xtb_energy(f"XTBREFINED_{self.mol_name}.xyz")
        except Exception as e:
            logging.error("Substrate: error in XTB._read_xtb_energy(): electronic energy could not be read from xtb output file: %s: %s", e.__class__.__name__, e)
        else:
            for atom_idx in self.info_dict["site_info"]:
                self._run_workflow(atom_idx)

        # Change back to the main directory
        os.chdir("../..")

        return self.feature_dict

    def _run_workflow(self, root_atom_idx: int) -> None:
        """Method for calculating the xtb features for a given site of the molecule as specified by root_atom_idx."""
        # CH-group-specific attributes
        self.root_atom_idx = root_atom_idx

        # Individual features
        primary_features = {"xtb_position_energy": self._get_xtb_position_energy}

        # Calculate features and store them in the feature dictionary
        for feature_name, method in primary_features.items():
            self._set_feature(feature_name, method)

    def _set_feature(self, feature_name: str, get_method: Callable[[], float]) -> None:
        """Method for writing the feature values to the feature dictionary."""
        try:
            result = get_method()
        except Exception as e:
            logging.error("Adduct-%s: error in XTB._set_feature(): %s could not be successfully calculated: %s: %s", self.root_atom_idx, feature_name, e.__class__.__name__, e)
        else:
            # Store the obtained energy for all symmetry equivalent positions
            for site_idx in self.info_dict["site_info"][str(self.root_atom_idx)]["site_indices"]:
                self.feature_dict[str(site_idx)][feature_name] = result

    def _get_xtb_position_energy(self) -> float:
        """Method for calculating the reaction energy between the substrate and the adduct."""
        eh_kj_mol = 2625.49964
        xtb_energy_et2o = -17.726984997786 * eh_kj_mol
        xtb_energy_tt_2plus = -35.445112772431 * eh_kj_mol
        xtb_energy_me_cation = -3.130475426030 * eh_kj_mol
        xtb_energy_tbu_cation = -12.766196633834 * eh_kj_mol

        xtb_energy_position_modified = self._read_xtb_energy(f"XTBREFINED_{self.mol_name}_adduct_Cpos-{self.root_atom_idx}.xyz")

        # Negative value of the reaction energy for the affinities
        if self.adduct_type == "proton":
            xtb_position_energy = -(xtb_energy_position_modified - self.substrate_xtb_energy)
        if self.adduct_type == "Me":
            xtb_position_energy = -(xtb_energy_position_modified - self.substrate_xtb_energy - xtb_energy_me_cation)
        if self.adduct_type == "tBu":
            xtb_position_energy = -(xtb_energy_position_modified - self.substrate_xtb_energy - xtb_energy_tbu_cation)
        if self.adduct_type == "TS":
            xtb_position_energy = xtb_energy_position_modified - self.substrate_xtb_energy - xtb_energy_et2o - xtb_energy_tt_2plus

        return xtb_position_energy

    @staticmethod
    def _read_xtb_energy(file_path: str) -> float:
        """Static method for extracting the electronic energy from an xtb output file."""
        eh_kj_mol = 2625.49964
        with open(file_path, "r") as f:
            second_line = f.readlines()[1]
        energy, _, _ = second_line.split("|")
        energy = float(energy.split()[-2]) * eh_kj_mol
        return energy


class XTBChargeFukui(MolPipeline):
    """A class for calculating atomic partial charges and Fukui coefficients with xtb. It inherits from MolPipeline to calculate the molecular charge and the number of unpaired electrons.

    Parameters
    ----------
    feature_dict : dict
        Feature dictionary with the atom indices as keys. The values are dictionaries with the feature name as keys and the numbers as values.
    
    Attributes
    ----------
    None

    Returns
    -------
    feature_dict : dict
        Feature dictionary updated with the new features.
    
    """

    def __init__(
        self,
        n_procs: int,
        compound_dir: str,
        info_dict: dict,
        mol_name: str,
        target_mol_charge: int,
        solvent: str,
        feature_dict: dict,
    ) -> None:
        super().__init__(n_procs, compound_dir, info_dict, mol_name, target_mol_charge, solvent)
        self.feature_dict = feature_dict

    def get_values(self) -> dict:
        """Method that executes the entire workflow."""
        # General attributes
        self.mol = self.info_dict["mol"]
        self.log = ""  # is is ignored in the end

        # Change to the folder of the compound
        os.chdir(self.compound_dir)

        # Run workflow
        self._get_charge()
        self._get_uhf()
        self._run_xtb_fukui_calculation()
        self._get_fukui_coefficients_and_charges()

        # Change back to the main directory and return updated feature dictionary
        os.chdir("../..")
        return self.feature_dict

    def _run_xtb_fukui_calculation(self) -> None:
        """Method that runs the calculation of the Fukui coefficients with xtb as a subprocess."""
        # Set up working directory
        working_dir = "work_xtb_fukui"
        if os.path.isdir(working_dir):
            shutil.rmtree(working_dir)
        os.mkdir(working_dir)

        # Copy input file to working directory
        shutil.copy2(f"XTBREFINED_{self.mol_name}.xyz", working_dir)

        # Change to working directory
        os.chdir("work_xtb_fukui")

        # Define xtb run command
        xtb_command = f"xtb XTBREFINED_{self.mol_name}.xyz --chrg {self.charge} --uhf {self.uhf}"

        # Add solvent if requested
        if self.solvent is not None:
             xtb_command += f" --alpb {self.solvent}"
        
        # Channel output to file
        xtb_command += " --vfukui > out.out"

        # Run xtb
        run_command = run(
            xtb_command,
            stdout=PIPE,
            stderr=PIPE,
            shell=True,
        )
        return_code = run_command.returncode
        stderr = str(run_command.stderr).split("'")[1].split("\\")[0]

        # Check for successful termination and get results
        if all(
            [
                stderr == "normal termination of xtb",
                return_code == 0,
                os.path.isfile("out.out"),
            ]
        ):
            # Copy xtb output file to the compound folder
            os.chdir("..")
            shutil.copy2(os.path.join(working_dir, "out.out"), f"XTBFUKUI_{self.mol_name}.out")
        else:
            os.chdir("..")
            logging.error("Substrate: error in XTBFukui._run_xtb_fukui_calculation(): the calculation of atomic charges and Fukui indices with xtb failed.")

        # Clean up
        shutil.rmtree(working_dir)
    
    def _get_fukui_coefficients_and_charges(self) -> None:
        """Method for extracting the calculated features from the xtb output file and for storing them in the feature dictionary."""
        # Open output file
        with open(f"XTBFUKUI_{self.mol_name}.out", "r") as f:
            fukui_out = f.readlines()
        
        # Find positions in the file where data is stored
        start_line_idx_charge = None
        start_line_idx_fukui = None
        for line_idx, line in enumerate(fukui_out):
            if line == "     #        f(+)     f(-)     f(0)\n":
                start_line_idx_fukui = line_idx
            if "     #   Z          covCN         q      C6AA" in line:
                start_line_idx_charge = line_idx
        
        if start_line_idx_charge is None or start_line_idx_fukui is None:
            logging.error("Substrate: error in XTBFukui._get_fukui_coefficients_and_charges(): the atomic charges and Fukui indices could not be extracted. Check the xtb output file.")
        else:
            # General information required for extracting the data
            number_of_atoms, c_positions, atom_list = self._get_atom_list_info(f"XTBREFINED_{self.mol_name}.xyz")

            # Loop over relevant part of the output file and extract the Fukui coefficients
            check_counter_fukui = 0
            for line in fukui_out[start_line_idx_fukui+1:start_line_idx_fukui+number_of_atoms+1]:
                splitted = line.split()
                for position in c_positions:
                    xtb_position = str(int(position)+1)  # xtb is 1-based, RDKit is 0-based
                    if splitted[0] == f"{xtb_position}{atom_list[position]}":
                        check_counter_fukui += 1
                        self.feature_dict[str(position)]["xtb_fukui_plus"] = float(splitted[2])
                        self.feature_dict[str(position)]["xtb_fukui_minus"] = float(splitted[1])
                        self.feature_dict[str(position)]["xtb_fukui_zero"] = float(splitted[3])
                        self.feature_dict[str(position)]["xtb_fukui_dual"] = float(splitted[2]) - float(splitted[1])
            
            # Double-check that Fukui indices were found for every atom
            if check_counter_fukui != len(c_positions):
                logging.error("Substrate: error in XTBFukui.get_fukui_coefficients_and_charges(): Fukui indices could not be extracted for every atom. Check the xtb output file.")

            # Loop over relevant part of the output file and extract the atomic partial charges
            check_counter_charge = 0
            for line in fukui_out[start_line_idx_charge+1:start_line_idx_charge+number_of_atoms+1]:
                splitted = line.split()
                for position in c_positions:
                    xtb_position = str(int(position)+1)  # xtb is 1-based, RDKit is 0-based
                    if splitted[0] == xtb_position:
                        check_counter_charge += 1
                        self.feature_dict[str(position)]["xtb_partial_charge"] = float(splitted[4])
            
            # Double-check that atomic charges were found for every atom
            if check_counter_charge != len(c_positions):
                logging.error("Substrate: error in XTBFukui._get_fukui_coefficients_and_charges(): Atomic charges could not be extracted for every atom. Check the xtb output file.")
    
    @staticmethod
    def _get_atom_list_info(xyz_file_path: str) -> tuple[int, list, list]:
        """Static method for extracting the number of atoms, the atom indices, and the atom symbols from an xyz file."""
        with open(xyz_file_path, "r") as f:
            xyz_file_content = f.readlines()
        number_of_atoms = int(xyz_file_content[0])
        c_positions = list(range(number_of_atoms))
        atom_list = []
        for line in xyz_file_content[2:]:
            atom_list.append(line.split()[0])
        return number_of_atoms, c_positions, atom_list
