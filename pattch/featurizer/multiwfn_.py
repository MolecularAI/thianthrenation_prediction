"""Feature calculation with Multiwfn."""

import os
import logging
from subprocess import PIPE, run


class AlieLea:
    """A class to calculate average local ionization energy (ALIE) and local electron affinity (LEA) values for individual atoms with multiWFN.

    Parameters
    ----------
    compound_dir : str
        Path to the folder of the currently treated molecule.
    feature_dict : dict
        Feature dictionary with the atom indices as keys. The values are dictionaries with the feature name as keys and the numbers as values.
    mol_name : str
        Name of the currently treated compound as specified by the user.

    Returns
    -------
    feature_dict : dict
        Feature dictionary updated with the new features.
    
    """

    def __init__(
            self,
            compound_dir: str,
            feature_dict: dict,
            mol_name: str,
            ) -> None:
        self.compound_dir = compound_dir
        self.feature_dict = feature_dict
        self.mol_name = mol_name

    def get_values(self) -> dict:
        """Method for getting the featuers."""
        # Change to the folder of the compound
        os.chdir(self.compound_dir)

        # Run workflow
        self._run_multiwfn()
        self._read_multiwfn_output_file_alie()
        self._read_multiwfn_output_file_lea()

        # Change back to the main directory
        os.chdir("../..")

        return self.feature_dict

    def _run_multiwfn(self) -> None:
        """Method for executing the multiWFN program to get the ALIE and LEA values."""
        # ALIE
        multiwfn_command = f'echo -e "12\n2\n2\n0\n11\nn\nq" | Multiwfn_noGUI molden.input > MULTIWFNALIE_{self.mol_name}.out'
        _ = run(
            multiwfn_command,
            stdout=PIPE,
            stderr=PIPE,
            shell=True,
        )

        # LEA
        multiwfn_command = f'echo -e "12\n2\n4\n0\n11\nn\nq" | Multiwfn_noGUI molden.input > MULTIWFNLEA_{self.mol_name}.out'
        _ = run(
            multiwfn_command,
            stdout=PIPE,
            stderr=PIPE,
            shell=True,
        )

    def _read_multiwfn_output_file_alie(self) -> None:
        """Method that reads the output file of the ALIE calculation and stores the results."""
        try:
            # Open output file
            with open(f"MULTIWFNALIE_{self.mol_name}.out", "r") as f:
                multiwfn_output = f.readlines()

            # Extract data
            for line_idx, line in enumerate(multiwfn_output):
                if line == " Minimal, maximal and average value are in eV, variance is in eV^2\n":
                    start_idx = line_idx + 2
                if line == " If outputting the surface facets to locsurf.pdb in current folder? By which you can visualize local surface via third-part visualization program such as VMD (y/n)\n":
                    end_idx = line_idx - 1

            # Save values to feature dictionary
            for line in multiwfn_output[start_idx:end_idx]:
                splitted = line.split()
                self.feature_dict[str(int(splitted[0]) - 1)]["multiwfn_min_alie"] = float(splitted[2])
        
        except Exception as e:
            logging.error("Substrate: error in AlieLea._read_multiwfn_output_file_alie(): calculation of ALIE values failed: %s: %s", e.__class__.__name__, e)

    def _read_multiwfn_output_file_lea(self) -> None:
        """Method that reads the output file of the LEA calculation and stores the results."""
        try:
            # Open output file
            with open(f"MULTIWFNLEA_{self.mol_name}.out", "r") as f:
                multiwfn_output = f.readlines()

            # Extract data
            for line_idx, line in enumerate(multiwfn_output):
                if line == " Note: Below minimal and maximal values are in eV\n":
                    start_idx = line_idx + 2
                if line == " Note: Average and variance below are in eV and eV^2 respectively\n":
                    end_idx = line_idx - 1

            # Save values to feature dictionary
            for line in multiwfn_output[start_idx:end_idx]:
                splitted = line.split()
                self.feature_dict[str(int(splitted[0]) - 1)]["multiwfn_max_lea"] = float(splitted[-1])

        except Exception as e:
            logging.error("Substrate: error in AlieLea._read_multiwfn_output_file_lea(): calculation of LEA values failed: %s: %s", e.__class__.__name__, e)
