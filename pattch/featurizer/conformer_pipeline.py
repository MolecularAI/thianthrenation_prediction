"""Generation of a 3D conformer ensemble prior to feature calculation."""

import os
import shutil
import logging
import numpy as np
from typing import Union
from datetime import datetime
from subprocess import PIPE, run
from concurrent.futures import ProcessPoolExecutor
import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdForceFieldHelpers
from rdkit.Chem.rdMolAlign import GetBestRMS


class MolPipeline:
    """A class that implements the conformer generation pipeline of the PATTCH workflow using RDKit and xtb.
    1) Generate an ensemble of conformers with RDKit.
    2) Optimize the conformers with MMFFs
    3) Calculate the structures' energy with xtb.
    4) Obtain the lowest energy structure and optimize it with xtb.

    Parameters
    ----------
    n_procs : int
        Number of processors used during the run.
    compound_dir : str
        Path to the folder of the currently treated molecule.
    info_dict : dict
        Dictionary with general information on the molecule (e.g., mol object, SMILES, potential sites of substitution).
    mol_name : str
        Name of the currently treated compound as specified by the user.
    target_mol_charge : int
        Target molecular charge of the compound.
    solvent : str
        Solvent to use for the xtb calculations
    site_idx : int
        Carbon atom index of the current C-H site. Is only specified if adducts are treated.
    
    Attributes
    ----------
    smiles : str
        SMILES string of the currently treated molecule.
    mol : rdkit.Chem.rdchem.Mol
        RDKit mol object of the currently treated molecule.
    constraints_dict : dict
        Dictionary for constraining internal coordinates (e.g., bond distances) if required. This remains empty within the PATTCH workflow.
    charge : int
        Molecular charge of the currently treated molecule.
    uhf : int
        Number of unpaired electrons of the currently treated molecule.
    error : bool
        For error handling within the workflow.
    conformer_data : dict
        Dictionary with all the information on the generated conformers.
    log : str
        Log string for keeping track of the workflow; gets printed to an output file in the end.
    self.N_CONF_MIN : int
        Minimum number of conformers to be generated by RDKit. The final number can be below this hyperparameter because of the RMSD filter.
    N_CONF_MAX : int
        Maximum number of conformers to be generated by RDKit.
    RDKIT_RANDOM_SEED : int
        Random seed for the conformer generation pipeline (required for reproducible) results.
    FF_OPT_N_ITER : int
        Maximum number of structure optimization steps for MMFFs structure optimizations. 
    RMSD_THRES : float
        RMSD threshold for rejecting a new conformer if it is to similar to an already existing one.
    N_CONF_XTB_OPT : int
        Number of conformers to be optimized by xtb.
    XTB_FORCE_CONST : float
        Force constant used for constrained structure optimizations with xtb.
    start_time : datetime.datetime
        Time stamp of when the conformer generation pipeline starts. Required for calculating the runtime.
    
    Returns
    -------
    None

    """

    def __init__(
        self,
        n_procs: int,
        compound_dir: str,
        info_dict: dict,
        mol_name: str,
        target_mol_charge: int,
        solvent: str,
        site_idx: int = None,
    ) -> None:
        self.n_procs = n_procs
        self.compound_dir = compound_dir
        self.info_dict = info_dict
        self.mol_name = mol_name
        self.target_mol_charge = target_mol_charge
        self.solvent = solvent
        self.site_idx = site_idx

        # General attributes
        self.smiles = None
        self.mol = None
        self.constraints_dict = None
        self.charge = None
        self.uhf = None
        self.error = False
        self.conformer_data = None
        self.log = None
        self.N_CONF_MIN = None
        self.N_CONF_MAX = None
        self.RDKIT_RANDOM_SEED = None
        self.FF_OPT_N_ITER = None
        self.RMSD_THRES = None
        self.N_CONF_XTB_OPT = None
        self.XTB_FORCE_CONST = None
        self.start_time = None

    def search_conformers(self) -> None:
        """Method that executes the entire pipeline."""
        self.smiles = self.info_dict["smiles"]
        self.mol = self.info_dict["mol"]
        self.constraints_dict = self.info_dict["constraints"]
        
        # Results dictionary
        self.conformer_data = {}

        # Separate small log file for the conformer generation workflow
        self.log = "-----------------------------------------------------------------\n Conformer generation with RDKit and xtb for the PATTCH workflow \n-----------------------------------------------------------------\n\n"
        self.log += f"Molecule name: {self.mol_name}\n"

        if self.site_idx is None:
            self.log += "Molecule type: substrate\n"
        else:
            self.log += "Molecule type: adduct\n"

        self.log += f"Molecule smiles: {self.smiles}\n"
        self.log += (f"Predifined target charge of the molecule: {self.target_mol_charge}\n\n")

        # Hyperparameter
        self.N_CONF_MIN = 50
        self.N_CONF_MAX = 1000
        self.RDKIT_RANDOM_SEED = 42
        self.FF_OPT_N_ITER = 300
        self.RMSD_THRES = 0.1
        self.N_CONF_XTB_OPT = 1
        self.XTB_FORCE_CONST = 1.0

        # Change to the folder of the compound
        os.chdir(self.compound_dir)

        # Do conformational searching
        self.start_time = datetime.now()

        # Execute the workflow
        self._get_charge()
        self._get_uhf()
        self._get_conformer_ensemble()
        if self.error is False:
            self._get_gfn2_conformer_energies()
            self._filter_conformers()
            self._get_gfn2_structures_and_energies()
            self._sort_conformers()
        
        # Write short output file of the entire process
        self._write_log_file()

        # Change back to the main directory
        os.chdir("../..")

    def _get_charge(self) -> None:
        """Method that determines the total charge of the molecule."""
        self.charge = Chem.GetFormalCharge(self.mol)

        if self.charge == self.target_mol_charge:
            self.log += f"Molecular charge was determined to {self.charge} (matches predifined target charge).\n"
        else:
            self.log += f"Molecular charge was determined to {self.charge} (does not match predifined target charge).\n"
            if self.site_idx is None:
                logging.warning("Substrate: warning in MolPipeline._get_charge(): the charge of the molecule (%s) does not match the expected charge (%s).", self.charge, self.target_mol_charge)
            else:
                logging.warning("Adduct-%s: warning in MolPipeline._get_charge(): the charge of the molecule (%s) does not match the expected charge (%s).", self.site_idx, self.charge, self.target_mol_charge)

    def _get_uhf(self) -> None:
        """Method that determines the number of unpaired electrons of the molecule."""
        self.uhf = 0
        for atom in self.mol.GetAtoms():
            self.uhf += atom.GetNumRadicalElectrons()
        self.log += f"Number of unpaired electrons was determined to {self.uhf}.\n\n"

        if self.uhf != 0:
            if self.site_idx is None:
                logging.warning("Substrate: warning in MolPipeline._get_uhf(): the molecule has unpaired electrons.")
            else:
                logging.warning(
                    "Adduct-%s: warning in MolPipeline._get_uhf(): the molecule has unpaired electrons.", self.site_idx)

    def _get_conformer_ensemble(self) -> None:
        """Method that generates a conformer ensemble with RDKit."""
        # Numer of conformers to generate
        n_conf = int(np.clip(3 ** AllChem.CalcNumRotatableBonds(Chem.MolFromSmiles(self.smiles)), self.N_CONF_MIN, self.N_CONF_MAX))

        # Try embedding conformers
        try:
            AllChem.EmbedMultipleConfs(
                self.mol,
                numConfs=n_conf,
                pruneRmsThresh=self.RMSD_THRES,
                randomSeed=self.RDKIT_RANDOM_SEED,
                useExpTorsionAnglePrefs=True,
                useBasicKnowledge=True,
                numThreads=self.n_procs,
            )
        except Exception as e:
            self.log += "ERROR: RDKit conformer embedding failed.\n"
            self.error = True
            if self.site_idx is None:
                logging.error("Substrate: error in MolPipeline._get_conformer_ensemble(): RDKit conformer embedding failed: %s: %s", e.__class__.__name__, e)
            else:
                logging.error("Adduct-%s: error in MolPipeline._get_conformer_ensemble(): RDKit conformer embedding failed: %s: %s", self.site_idx, e.__class__.__name__, e)
        else:
            # Get the force field for optimization (including potential constraints)
            force_field = self._get_force_field()

            # Optimize all conformers
            AllChem.OptimizeMoleculeConfs(
                self.mol,
                force_field,
                maxIters=self.FF_OPT_N_ITER,
                numThreads=self.n_procs,
            )

            # Format results
            atom_symbols = [atom.GetSymbol() for atom in self.mol.GetAtoms()]
            for conf in self.mol.GetConformers():
                coords = str(len(atom_symbols)) + "\n\n"
                for idx, line in enumerate(conf.GetPositions()):
                    coords += f"  {atom_symbols[idx]}    "
                    coords += "    ".join([str(round(x, 7)) for x in line])
                    coords += "\n"
                self.conformer_data[conf.GetId()] = {
                    "opt_mmffs_xyz": coords,
                    "opt_mmffs_xtb_energy": None,
                    "opt_xtb_xyz": None,
                    "opt_xtb_energy": None,
                    "selected_for_xtb_opt": False,
                }

            # Add information to log
            self.log += f"{self.N_CONF_MAX} conformers are generated at maximum (minimum conformer generation attempts: {self.N_CONF_MIN}).\n"
            self.log += f"{len(self.conformer_data)} conformers were generated by RDKit (RMSD threshold: {self.RMSD_THRES} A).\n"

            # PATTCH logging
            if self.site_idx is None:
                logging.info("Substrate: conformer ensemble generated.")
            else:
                logging.info("Adduct-%s: conformer ensemble generated.", self.site_idx)

    def _get_gfn2_conformer_energies(self) -> None:
        """Method that calculates the GFN2-xTB energy (on the MMFFs structures) of all conformers in parallel."""
        # Format input
        all_xtb_inputs = [
            {
                "conf_idx": conf_idx,
                "xyz": conf_data["opt_mmffs_xyz"],
                "run_type": "sp",
                "constraints": None,
                "run_xtb_for_conf": True,
            }
            for conf_idx, conf_data in self.conformer_data.items()
        ]

        # Set up process pool executor for xtb jobs
        with ProcessPoolExecutor(max_workers=self.n_procs) as executer:
            results = executer.map(self._run_xtb_calculation, all_xtb_inputs)

        # Get results and save them
        for result in results:
            self.conformer_data[result[0]]["opt_mmffs_xtb_energy"] = result[1]["energy"]

        # Sort conformer dictionary according to the xtb energy of the MMFFs structures
        self.conformer_data = dict(sorted(self.conformer_data.items(), key=lambda x: x[1]["opt_mmffs_xtb_energy"], reverse=False))

        self.log += "GFN2-xTB energies of all conformers were calculated.\n\n"

    def _filter_conformers(self) -> None:
        """Method that selects the lowest n conformers (N_CONF_XTB_OPT) for further xtb optimization, including an RMSD threshold (RMSD_THRES)."""

        def check_potential_new_member(mol, already_selected_indices, potential_new_idx, thres):
            rmsd_values = []
            for conf_idx in already_selected_indices:
                rmsd_values.append(GetBestRMS(mol, mol, conf_idx, potential_new_idx))
            if min(rmsd_values) > thres:
                return True
            return False

        # Remove hydrogen atoms to make clustering faster
        self.mol = Chem.RemoveHs(self.mol)

        # Initialize selection
        selected_conf_indices = [list(self.conformer_data.keys())[0]]
        counter_added = 0
        counter_rejected = 0

        # Select the conformers considering the RMSD threshold
        for conf_idx in self.conformer_data:
            if conf_idx in selected_conf_indices:
                continue
            if counter_added >= self.N_CONF_XTB_OPT - 1:
                break

            check = check_potential_new_member(self.mol, selected_conf_indices, conf_idx, self.RMSD_THRES)
            if check is True:
                selected_conf_indices.append(conf_idx)
                counter_added += 1
            else:
                counter_rejected += 1

        # Mark all selected conformers for optimization
        for conf_idx in selected_conf_indices:
            self.conformer_data[conf_idx]["selected_for_xtb_opt"] = True

        self.log += f"{len(selected_conf_indices)} conformers were selected for optimization with xtb ({counter_rejected} conformers were rejected due to the RMSD threshold).\n\n"

    def _get_gfn2_structures_and_energies(self) -> None:
        """Method that calculates the GFN2-xTB structures and energies of all selected conformers in parallel."""
        # Format input
        all_xtb_inputs = [
            {
                "conf_idx": conf_idx,
                "xyz": conf_data["opt_mmffs_xyz"],
                "run_type": "opt",
                "run_xtb_for_conf": conf_data["selected_for_xtb_opt"],
                "constraints": self._generate_xtb_constraints_input(),
            }
            for conf_idx, conf_data in self.conformer_data.items()
        ]

        # Set up process pool executor for xtb jobs
        with ProcessPoolExecutor(max_workers=self.n_procs) as executer:
            results = executer.map(self._run_xtb_calculation, all_xtb_inputs)

        # Get results and save them
        for result in results:
            self.conformer_data[result[0]]["opt_xtb_xyz"] = result[1]["xyz"]
            self.conformer_data[result[0]]["opt_xtb_energy"] = result[1]["energy"]

    def _sort_conformers(self) -> None:
        """Method that selects the lowest energy conformer."""
        # Sort conformers
        conformer_data_opt_mmffs_xtb_energy = dict(sorted(self.conformer_data.items(), key=lambda x: x[1]["opt_mmffs_xtb_energy"], reverse=False))
        conformer_data_opt_xtb_energy = dict(sorted(self.conformer_data.items(), key=lambda x: x[1]["opt_xtb_energy"], reverse=False))
        
        best_conf_idx = list(conformer_data_opt_xtb_energy.keys())[0]

        # Formatting
        ensemble_mmffs = ""
        for conf_idx, conf_data in conformer_data_opt_mmffs_xtb_energy.items():
            formatted_mmffs = conf_data["opt_mmffs_xyz"].replace("\n\n", f"\nGFN2-xTB energy: {conf_data['opt_mmffs_xtb_energy']} Eh | solvent: {self.solvent} | RDKit conf. idx: {conf_idx}\n")
            ensemble_mmffs += formatted_mmffs
            conf_data["opt_mmffs_xyz"] = formatted_mmffs

        # Formatting
        ensemble_xtb = ""
        counter = 0
        for conf_idx, conf_data in conformer_data_opt_xtb_energy.items():
            if conf_data["opt_xtb_energy"] != 1:
                counter += 1
                formatted_xtb = conf_data["opt_xtb_xyz"].replace("\n\n", f"\nGFN2-xTB energy: {conf_data['opt_xtb_energy']} Eh | solvent: {self.solvent} | RDKit conf. idx: {conf_idx}\n")
                ensemble_xtb += formatted_xtb
                conf_data["opt_xtb_xyz"] = formatted_xtb

        # Write result files
        with open(f"ENSEMBLEALL_{self.mol_name}.xyz", "w") as f:
            f.write(ensemble_mmffs)
        with open(f"ENSEMBLEXTBOPT_{self.mol_name}.xyz", "w") as f:
            f.write(ensemble_xtb)
        with open(f"XTBREFINED_{self.mol_name}.xyz", "w") as f:
            f.write(conformer_data_opt_xtb_energy[best_conf_idx]["opt_xtb_xyz"])

        # Add information to log
        self.log += f"{counter} xtb optimizations were successfully performed.\n\n"
        self.log += "Conformational searching done.\n\n"

    def _write_log_file(self) -> None:
        """Method that writes the log file of the conformer pipeline and saves it."""
        self.log += "Hurray: All done.\n"
        self.log += f"Runtime: {datetime.now() - self.start_time}\n"
        with open(f"CONVSEARCH_{self.mol_name}.out", "w") as f:
            f.write(self.log)

    def _get_force_field(self) -> rdkit.ForceField.rdForceField.ForceField:
        """Method that sets up a force field for structure optimization."""
        # Get force field
        mmffps = rdForceFieldHelpers.MMFFGetMoleculeProperties(self.mol)
        force_field = rdForceFieldHelpers.MMFFGetMoleculeForceField(self.mol, mmffps)

        # Add bond constraints
        if "distances" in self.constraints_dict:
            for bond in self.constraints_dict["distances"]:
                force_field.MMFFAddDistanceConstraint(bond[0], bond[1], False, bond[2], bond[2], 1.0e10)

        return force_field

    def _run_xtb_calculation(self, xtb_input_dict: dict) -> tuple[int, dict]:
        """Method that runs an xtb calculation as a subprocess.
        xtb_input_dict must have "conf_idx", "xyz", "run_type", "constraints", and "run_xtb_for_conf" as keys.
        """
        # Initialize output
        xtb_output_data = {"energy": 1, "gibbs_free_energy": 1, "xyz": None}

        # Don't run xtb in case the structure was not selected for optimization
        if xtb_input_dict["run_xtb_for_conf"] is False:
            return (xtb_input_dict["conf_idx"], xtb_output_data)

        # Set up working directory and create structure input file
        working_dir = f"work_{xtb_input_dict['conf_idx']}"
        if os.path.isdir(working_dir):
            shutil.rmtree(working_dir)
        os.mkdir(working_dir)

        with open(os.path.join(working_dir, "struc.xyz"), "w") as f:
            f.write(xtb_input_dict["xyz"])

        # Change to working directory
        os.chdir(working_dir)

        # Define xtb run command
        xtb_command = f"xtb struc.xyz --chrg {self.charge} --uhf {self.uhf}"

        # Check type of calculation (single point: no additional flag, optimization, frequency calc. or opt followed by freq calculation)
        if xtb_input_dict["run_type"] == "opt":
            xtb_command += " --opt"
        if xtb_input_dict["run_type"] == "freq":
            xtb_command += " --hess"
        if xtb_input_dict["run_type"] == "opt_freq":
            xtb_command += " --ohess"

        # Add solvent if requested
        if self.solvent is not None:
            xtb_command += f" --alpb {self.solvent}"

        # Add constraints for optimization if requested
        if xtb_input_dict["constraints"] != None:
            xtb_command += " --input constraints.input"
            with open("constraints.input", "w") as f:
                f.write(xtb_input_dict["constraints"])

        # Add molden flag for the final structure optimization of the substrate
        if self.site_idx is None and xtb_input_dict["run_type"] == "opt":
            xtb_command += " --molden"

        # Channel output to file
        xtb_command += " > out.out"

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
        if all([
            stderr == "normal termination of xtb",
            return_code == 0,
            os.path.isfile("out.out"),
            ]):
            # Get xtb energy
            xtb_output_data = self._read_xtb_output()
            os.chdir("..")

            # Copy xtb output files of the final structure optimization and the molden.input file in case of the substrate to the compound folder
            if self.site_idx is None and xtb_input_dict["run_type"] == "opt":
                shutil.copy2(os.path.join(working_dir, "out.out"), f"XTBREFINED_{self.mol_name}.out")
                shutil.copy2(os.path.join(working_dir, "molden.input"), "molden.input")
            elif xtb_input_dict["run_type"] == "opt":
                shutil.copy2(os.path.join(working_dir, "out.out"), f"XTBREFINED_{self.mol_name}.out")

        else:
            os.chdir("..")
            run_type = xtb_input_dict["run_type"]

            # PATTCH logging for substrate
            if self.site_idx is None and run_type == "opt":
                logging.error("Substrate: error in MolPipeline._run_xtb_calculation(): xtb calculation failed for run type '%s'.", run_type)
            if self.site_idx is None and run_type == "sp":
                logging.warning("Substrate: warning in MolPipeline._run_xtb_calculation(): xtb calculation failed for run type '%s' for one of the evaluated conformers. The energy of this conformer is set to 1 (will be ignored).", run_type)
            
            # PATTCH logging for adduct
            if self.site_idx is not None and run_type == "opt":
                logging.error("Adduct-%s: error in MolPipeline._run_xtb_calculation(): xtb calculation failed for run type '%s'.", self.site_idx, run_type)
            if self.site_idx is not None and run_type == "sp":
                logging.warning("Adduct-%s: warning in MolPipeline._run_xtb_calculation(): xtb calculation failed for run type '%s' for one of the evaluated conformers. The energy of this conformer is set to 1 (will be ignored).", self.site_idx, run_type)

        # Clean up and return output information
        shutil.rmtree(working_dir)
        return (xtb_input_dict["conf_idx"], xtb_output_data)

    def _read_xtb_output(self) -> dict:
        """Method that extracts the electronic energy from an xtb output file."""
        # Define data structure
        xtb_output_data = {"energy": 1, "gibbs_free_energy": 1, "xyz": None}

        # Open output file
        with open("out.out", "r") as f:
            xtb_output = f.readlines()

        if os.path.isfile("xtbopt.xyz"):
            with open("xtbopt.xyz", "r") as f:
                xyz = f.readlines()
            xyz[1] = "\n"
            xtb_output_data["xyz"] = "".join(xyz)

        # Read output file
        for line in xtb_output:
            if "TOTAL ENERGY" in line:
                xtb_output_data["energy"] = float(line.split()[-3])
            if "TOTAL FREE ENERGY" in line:
                xtb_output_data["gibbs_free_energy"] = float(line.split()[-3])
        return xtb_output_data

    def _generate_xtb_constraints_input(self) -> Union[str, None]:
        """Method that writes information on constraints (bonds, angles, dihedrales) in the xtb format."""
        if self.constraints_dict != {}:
            # Initialize constraints string
            constraints_string = f"$constrain\nforce constant={self.XTB_FORCE_CONST}\n"

            # Add bond constraints
            if "distances" in self.constraints_dict:
                for bond in self.constraints_dict["distances"]:
                    constraints_string += f"    distance: {bond[0]+1}, {bond[1]+1}, auto\n"

            # Add end of file marker
            constraints_string += "$end\n"
            return constraints_string
        return None
