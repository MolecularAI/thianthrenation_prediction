"""Main implementation of the PATTCH workflow, which is called through the predict method of the PATTCH class."""

import os
import shutil
import pickle
import logging
import pandas as pd
from collections import Counter
from datetime import datetime
from rdkit import Chem
from qmdesc import ReactivityDescriptorHandler

from .featurizer.conformer_pipeline import MolPipeline
from .featurizer.qmdesc_ import get_qmdesc_values
from .featurizer.multiwfn_ import AlieLea
from .featurizer.morfeus_rdkit import MorfeusRDKit
from .featurizer.xtb_ import XTB, XTBChargeFukui
from .featurizer.ortho_meta_para import OrthoMetaPara
from .featurizer.finalization import get_effective_features, adjust_features, add_categorical_feature

from .utils.preprocessor import get_mol_object, Preprocessor
from .utils.xyz import CheckConnectivity

from .model.prediction import run_classifier
from .model.visualization import make_single_image, make_grid_image


class PATTCH:
    """A class that implements the workflow of the Predictive model for the Aromatic ThianThrenation reaction for C-H functionalization (PATTCH).

    Parameters
    ----------
    n_procs : int
        Number of processors used during the run.
    adduct_type : str
        Defines which adduct is used for probing the individual C-H positions. Must be one of ['tBu', 'Me', 'proton', 'TS'].
    calc_xtb_fukuis : bool
        Use xtb to calculate Fukui coefficients in addition to qmdesc. The xtb Fukui coefficients are not used in the classification algorithm.

    Attributes
    ----------
    model : xgboost.sklearn.XGBClassifier
        Classification model, read from pickle file.
    pipeline : pycaret.internal.pipeline.Pipeline
        Data preprocessing pipeline for the classification model, read from pickle file.
    results_dict : dict
        Overall dictionary for storing the results and outputs of the workflow.
    qmdesc_handler : qmdesc.handler.ReactivityDescriptorHandler
        Handler object to make predictions with qmdesc.
    
    Returns
    -------
    None

    """

    def __init__(
        self,
        n_procs: int,
        adduct_type: str,
        calc_xtb_fukuis: bool,
        ) -> None:
        self.n_procs = n_procs
        self.adduct_type = adduct_type
        self.calc_xtb_fukuis = calc_xtb_fukuis

        # General attributes
        self.model = None
        self.pipeline = None

        # Initialize the overall results dictionary of the run
        self.results_dict = {}

        # Initialize the handler object of qmdesc
        self.qmdesc_handler = ReactivityDescriptorHandler()

    def predict(self, input_file_path: str) -> None:
        """Method that implements the PATTCH workflow. This method gets called by the user.
        
        Parameters
        ----------
        input_file_path : str
            Path to the input .csv file for the PATTCH workflow. Required columns are 'substrate_smiles' and 'substrate_name'.
        
        Returns
        -------
        None
        
        """
        self._read_input(input_file_path)
        self._run_precheck()
        if self.adduct_type == "tBu":
            self._load_model()
        self._run_pattch_workflow()
        self._clean_up()

    def _read_input(self, input_file_path: str) -> None:
        """Method for reading the input csv file and check if both required columns are present. Required columns are 'substrate_smiles' and 'substrate_name'."""
        # Read input file
        df = pd.read_csv(input_file_path)

        # Check if required input columns are present in the input files
        column_names = [n.lower() for n in list(df.columns)]
        if ("substrate_smiles" not in column_names or "substrate_name" not in column_names):
            raise ValueError("Error in PATTCH._read_input(): required input columns ('substrate_name' and 'substrate_smiles') were not found in input .csv file.")

        df.columns = column_names
        df = df.rename(columns={"substrate_smiles": "smiles"})
        df = df[["substrate_name", "smiles"]]

        # Check if substrate_name column has only unique entries
        duplicates = []
        for substrate_name, count in dict(Counter(df["substrate_name"])).items():
            if count > 1:
                duplicates.append(substrate_name)
        if len(duplicates) > 0:
            raise ValueError(f"Error in PATTCH._read_input(): duplicates were found in the 'substrate_name' column of the input .csv file: {duplicates}.")

        # Format results dictionary
        self.results_dict = df.set_index("substrate_name").to_dict(orient="index")
        self.results_dict = {name: {"substrate_info": data}for name, data in self.results_dict.items()}

        print("Input file was successfully read.")

    def _run_precheck(self) -> None:
        """Method for checking the directory specified by the user for already existing output folders and remove them."""
        if os.path.exists("output_files"):
            shutil.rmtree("output_files")
        os.mkdir("output_files")

        if os.path.exists("prediction_results"):
            shutil.rmtree("prediction_results")

        # Create prediction results folder in case tBu was specified as adduct type
        if self.adduct_type == "tBu":
            os.mkdir("prediction_results")
            print("Output and prediction results directory created.")
            print("Specified adduct type: tBu --> reactivity classifier will be executed.")
        else:
            print("Output directory created.")
            print(f"Specified adduct type: '{self.adduct_type}' --> reactivity classifier will NOT be executed. Only the features are calculated.\n")

        # Copy readme file to ouput files folder
        module_path = os.path.split(__file__)[0]
        shutil.copy2(os.path.join(module_path, "model", "_readme_files.txt"), os.path.join("output_files", "_readme.txt"))

    def _run_pattch_workflow(self) -> None:
        """Method for running the PATTCH workflow for a batch of molecules."""
        # General settings and data
        solvent = "acetonitrile"
        counter = 0
        feature_data_frames = []
        result_images = []

        # Loop over the individual molecules
        print(f"Running PATTCH for {len(self.results_dict)} compounds ...")
        for substrate_name, data in self.results_dict.items():
            counter += 1
            start_time = datetime.now()
            print(f"    > {substrate_name} ({counter}/{len(self.results_dict)}) ...", end=" ")

            # Make molecule folder for output files
            mol_folder_path = os.path.join("output_files", substrate_name)
            if os.path.exists(mol_folder_path):
                shutil.rmtree(mol_folder_path)
            os.mkdir(mol_folder_path)

            # Initialize logging
            log_file_path = os.path.join(mol_folder_path, f"PATTCH_{substrate_name}.log")
            if os.path.isfile(log_file_path):
                os.remove(log_file_path)

            logging.basicConfig(
                filename=log_file_path,
                level=logging.DEBUG,
                format="%(asctime)s - %(levelname)s - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
            logging.info("Starting PATTCH model workflow for %s.", substrate_name)

            # Get the mol object of the substrate
            substrate_mol, substrate_smiles = get_mol_object(data["substrate_info"]["smiles"])

            # Skip the molecule if no RDKit mol object could be generated for the substrate
            if substrate_mol is None or isinstance(substrate_mol, Chem.rdchem.Mol) is False:
                logging.error("Substrate: error in preprocessor.get_mol_object(): RDKit mol object could not be generated.")
                continue
            logging.info("Substrate: RDKit mol object generated.")

            # Set up further information on molecule
            data["substrate_info"]["smiles"] = substrate_smiles
            data["substrate_info"]["mol"] = substrate_mol
            data["substrate_info"]["constraints"] = {}

            # Do preprocessing
            self._run_preprocessor(substrate_name=substrate_name, data=data)

            # Get 3D structure of substrate with RDKit an xtb
            logging.info("Substrate: running conformer pipeline.")
            mp = MolPipeline(
                n_procs=self.n_procs,
                compound_dir=os.path.join("output_files", substrate_name),
                info_dict=data["substrate_info"],
                mol_name=substrate_name,
                target_mol_charge=0,
                solvent=solvent,
            )
            mp.search_conformers()

            # Check atom connectivity after structure optimization for the substrate
            conn = CheckConnectivity(
                mol=data["substrate_info"]["mol"],
                xyz_file_path=os.path.join("output_files", substrate_name, f"XTBREFINED_{substrate_name}.xyz"),
            )

            if conn.check() is False:
                logging.error("Substrate: error in xyz.CheckConnectivity(): atom connectivity determined after structure optimization does not match the expected atom connectivity from SMILES.")
                continue
            logging.info("Substrate: atom connectivity determined after structure optimization matches the expected atom connectivity from SMILES.")

            # Get 3D structures of the adducts with RDKit and xtb
            if self.adduct_type == "TS":
                target_mol_charge = 2
            else:
                target_mol_charge = 1
            for site_idx, site_data in data["site_info"].items():
                logging.info("Adduct-%s: running conformer pipeline.", site_idx)
                mp = MolPipeline(
                    n_procs=self.n_procs,
                    compound_dir=os.path.join("output_files", substrate_name),
                    info_dict=site_data,
                    mol_name=f"{substrate_name}_adduct_Cpos-{site_idx}",
                    target_mol_charge=target_mol_charge,
                    solvent=solvent,
                    site_idx=site_idx,
                )
                mp.search_conformers()

                # Check atom connectivity after structure optimization for the adducts (not possible for the approximate transition structures)
                if self.adduct_type != "TS":
                    conn = CheckConnectivity(
                        mol=site_data["mol"],
                        xyz_file_path=os.path.join("output_files", substrate_name, f"XTBREFINED_{substrate_name}_adduct_Cpos-{site_idx}.xyz"),
                    )
                    if conn.check() is False:
                        logging.warning("Adduct-%s: warning in xyz.CheckConnectivity(): atom connectivity of final structure changed after structure optimization.", site_idx)
                    else:
                        logging.info("Adduct-%s: atom connectivity of final structure matches the expected atom connectivity from SMILES.", site_idx)
                else:
                    logging.warning("Adduct-%s: warning in xyz.CheckConnectivity(): atom connectivity is not checked after approximate (constrained) transition structure optimization.", site_idx)

            # Calculate features
            logging.info("Calculating features.")
            feature_df = self._get_features(substrate_name=substrate_name, data=data)
            logging.info("Feature calculation completed.")

            # Save features and general information dictionary
            p = os.path.join("output_files", substrate_name, f"PATTCH_{substrate_name}")
            feature_df.to_csv(f"{p}.csv")
            feature_data_frames.append(feature_df)

            with open(f"{p}.pkl", "wb") as f:
                pickle.dump(data, f)

            if self.adduct_type == "tBu":
                # Make predictions
                logging.info("Running classification model.")
                results_df = run_classifier(
                    model=self.model,
                    pipeline=self.pipeline,
                    feature_df=feature_df,
                )
                logging.info("Done.")

                # Save results
                logging.info("Saving prediction results.")
                p = os.path.join("prediction_results", substrate_name)
                results_df.to_csv(f"{p}.csv")
                image = make_single_image(
                    results_df=results_df,
                    info_dict=data,
                    substrate_name=substrate_name,
                )
                result_images.append(image)
                logging.info("Done.")

            # Reset the logging configuration for the next iteration
            logging.info("Requested PATTCH workflow was fully executed.")
            logging.shutdown()
            logging.getLogger().handlers.clear()

            # Print run time per molecule
            end_time = datetime.now()
            run_time = end_time - start_time
            run_time = round(run_time.total_seconds(), 1)

            print(f"Done ({run_time} seconds).")

        if self.adduct_type == "tBu":
            # Generate a grid image of all prediction results
            if len(result_images) > 0:
                make_grid_image(image_list=result_images)

            # Save all calculated features for all molecules in one csv file
            if len(feature_data_frames) > 0:
                total_df = pd.concat(feature_data_frames)
                total_df.to_csv("prediction_results/_all.csv")

    def _run_preprocessor(self, substrate_name: str, data: dict) -> None:
        """Method for the preprocessing pipeline of the PATTCH workflow."""
        # Instantiate preprocessor
        pp = Preprocessor(info_dict=data)

        # Identify the sites that can undergo thianthrenation
        pp.get_substitution_sites()
        pp.get_unique_substitution_sites()
        logging.info("Substrate: potential sites for substitution (%s) identified.", list(pp.unique_substitution_sites.keys()))

        # Store results and further set up results dictionary
        self.results_dict[substrate_name]["site_info"] = {
            str(idx): {
                "site_indices": pp.unique_substitution_sites[idx],
                "smiles": None,
                "mol": None,
                "constraints": None,
                "ortho_atom_indices": {},
                "meta_atom_indices": {},
                "para_atom_indices": {},
            }
            for idx in pp.unique_substitution_sites
        }

        # Loop over the potential sites of substitution and build the adducts
        for site_idx in pp.unique_substitution_sites:
            pp.build_adduct(
                substitution_atom_idx=site_idx,
                adduct_type=self.adduct_type,
                )
            self.results_dict[substrate_name]["site_info"][str(site_idx)]["smiles"] = pp.adduct_smiles
            self.results_dict[substrate_name]["site_info"][str(site_idx)]["mol"] = pp.adduct_mol
            self.results_dict[substrate_name]["site_info"][str(site_idx)]["constraints"] = pp.adduct_constraints

    def _get_features(self, substrate_name: str, data: dict) -> pd.DataFrame:
        """Method for calculating all features for all relevant sites of a molecule."""
        # Feature dictionary for the individual sites of the molecule
        feature_dict = {str(atom.GetIdx()): {} for atom in data["substrate_info"]["mol"].GetAtoms()}

        # General information on the molecule
        substrate_smiles = data["substrate_info"]["smiles"]
        all_relevant_atom_indices = self._get_all_relevant_atom_indices(data)

        # Get xtb features
        logging.info("Calculating xtb features (position energy).")
        x = XTB(
            compound_dir=os.path.join("output_files", substrate_name),
            info_dict=data,
            feature_dict=feature_dict,
            mol_name=substrate_name,
            adduct_type=self.adduct_type,
        )
        feature_dict = x.get_values()
        logging.info("Done.")

        # Calculate xtb atomic partial charges and Fukui coefficients if requested
        if self.calc_xtb_fukuis is True:
            logging.info("Calculating xtb features (atomic charges and Fukui coefficients).")
            xfc = XTBChargeFukui(
                n_procs=None,  # not implemented for this class, no parallelization required here
                compound_dir=os.path.join("output_files", substrate_name),
                info_dict=data["substrate_info"],
                mol_name=substrate_name,
                target_mol_charge=0,
                solvent="acetonitrile",
                feature_dict=feature_dict,
            )
            feature_dict = xfc.get_values()
            logging.info("Done.")

        # Get multiWFN features
        logging.info("Calculating multiWFN features.")
        al = AlieLea(
            compound_dir=os.path.join("output_files", substrate_name),
            feature_dict=feature_dict,
            mol_name=substrate_name,
        )
        feature_dict = al.get_values()
        logging.info("Done.")

        # Get qmdesc features
        logging.info("Calculating qmdesc features.")
        try:
            feature_dict = get_qmdesc_values(
                handler=self.qmdesc_handler,
                substrate_smiles=substrate_smiles,
                feature_dict=feature_dict,
            )
        except Exception as e:
            logging.error("Error in PATTCH._get_features(): Calculating qmdesc features failed: %s: %s", e.__class__.__name__, e)
        logging.info("Done.")

        # Get morfeus and RDKit features
        logging.info("Calculating morfeus and RDKit features.")
        mr = MorfeusRDKit(
            compound_dir=os.path.join("output_files", substrate_name),
            info_dict=data,
            feature_dict=feature_dict,
            mol_name=substrate_name,
            atom_indices=all_relevant_atom_indices,
        )
        feature_dict = mr.get_values()
        logging.info("Done.")

        # Get features for the ortho-, meta-, and para-position
        logging.info("Calculating features for the ortho, meta, and para position.")
        try:
            omp = OrthoMetaPara(
                info_dict=data["site_info"],
                calc_xtb_fukuis=self.calc_xtb_fukuis,
                )
            feature_dict = omp.get_values(feature_dict=feature_dict, site_label="ortho")
            feature_dict = omp.get_values(feature_dict=feature_dict, site_label="meta")
            feature_dict = omp.get_values(feature_dict=feature_dict, site_label="para")
        except Exception as e:
            logging.error("Error in PATTCH._get_features(): Calculating features for ortho/meta/para-positions failed: %s: %s", e.__class__.__name__, e)
        logging.info("Done.")

        # Average the features between symmetry-equivalent positions
        logging.info("Calculating mean features for symmetry-equivalent positions.")
        try:
            feature_dict = get_effective_features(
                feature_dict=feature_dict,
                info_dict=data,
            )
        except Exception as e:
            logging.error("Error in PATTCH._get_features(): Calculating features for symmetry-equivalent positions failed: %s: %s", e.__class__.__name__, e)
        logging.info("Done.")

        # Calculate sterically-adjusted features
        logging.info("Calculating sterically adjusted features.")
        try:
            feature_dict = adjust_features(
                feature_dict=feature_dict,
                adduct_type=self.adduct_type,
                calc_xtb_fukuis=self.calc_xtb_fukuis
            )
        except Exception as e:
            logging.error("Error in PATTCH._get_features(): Calculating sterically-adjusted features failed: %s: %s", e.__class__.__name__, e)
        logging.info("Done.")

        # Add categorical features
        logging.info("Calculating categorical features.")
        try:
            # Differentiate between the transition state as adduct type (the lower the better) and the affinity values (the higher the better)
            if self.adduct_type == "TS":
                feature_dict = add_categorical_feature(feature_dict, "xtb_position_energy", "lowest")
            else:
                feature_dict = add_categorical_feature(feature_dict, "xtb_position_energy", "highest")
            
            # Only add the categorical feature for the xtb fukui_minus coefficient if the xtb Fukui coefficients were calculated
            if self.calc_xtb_fukuis is True:
                feature_dict = add_categorical_feature(feature_dict, "xtb_fukui_minus", "highest")

            # Add the remaining categorical features
            feature_dict = add_categorical_feature(feature_dict, "morfeus_buried_volume", "lowest")
            feature_dict = add_categorical_feature(feature_dict, "multiwfn_min_alie", "lowest")
            feature_dict = add_categorical_feature(feature_dict, "qmdesc_fukui_minus", "highest")

        except Exception as e:
            logging.error("Error in PATTCH._get_features(): Calculating categorical features failed: %s: %s", e.__class__.__name__, e)
        logging.info("Done.")

        # Add atom index for further processing
        for atom_idx, features in feature_dict.items():
            features["_site_idx"] = atom_idx

        # Finalize and return feature data frame
        feature_dict = {f"{substrate_name}_Cpos-{atom_idx}": features for atom_idx, features in feature_dict.items()}

        # Add SMILES string for further processing
        for identifier in feature_dict:
            feature_dict[identifier]["_substrate_smiles"] = substrate_smiles

        feature_df = pd.DataFrame(feature_dict).T
        return feature_df

    def _load_model(self) -> None:
        """Method for loading the classifier model."""
        # Path to the model folder within the module
        module_path = os.path.split(__file__)[0]

        # Load model
        with open(os.path.join(module_path, "model", "model.pkl"),"rb") as f:
            self.model = pickle.load(f)

        # Load pipeline
        with open(os.path.join(module_path, "model", "model_pipeline.pkl"), "rb") as f:
            self.pipeline = pickle.load(f)

        # Copy readme file to prediction results folder
        shutil.copy2(os.path.join(module_path, "model", "_readme_vis.txt"), os.path.join("prediction_results", "_readme.txt"))

        print("Data preprocessing pipeline and classification model loaded.\n")

    def _clean_up(self) -> None:
        """Method for doing the final clean-up after the run is completed."""
        # Remove pycarret log file
        if os.path.isfile("logs.log"):
            os.remove("logs.log")

    @staticmethod
    def _get_all_relevant_atom_indices(data: dict) -> list:
        """Static method for getting all relevant atom indices (all CH positions of a molecule)."""
        all_indices = []
        for site_data in data["site_info"].values():
            all_indices.extend(site_data["site_indices"])
        all_indices.sort()
        return all_indices
