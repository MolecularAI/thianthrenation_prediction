"""Extraction of the features for the ortho-, meta-, and para-position relative to a given C-H group."""

import logging
import numpy as np

class OrthoMetaPara:
    """A class for calculating the ortho, meta, and para features relative to a given C-H group.

    Parameters
    ----------
    info_dict : dict
        Dictionary with general information on the molecule (e.g., mol object, SMILES, potential sites of substitution).
    calc_xtb_fukuis : bool
        Defines whether xtb Fukui coefficients have been calculated or not.
    
    Attributes
    ----------
    feature_dict : dict
        Feature dictionary with the atom indices as keys. The values are dictionaries with the feature name as keys and the numbers as values.
    target_features : list
        List of features to calculate for the individual relative positions.
    
    Returns
    -------
    feature_dict : dict
        Feature dictionary updated with the new features.
    
    """
    def __init__(self, info_dict: dict, calc_xtb_fukuis: bool) -> None:
        self.info_dict = info_dict
        
        # General attributes
        self.feature_dict = None
        
        # Features that should be fetched from the ortho/meta/para sites
        self.target_features = [
            "multiwfn_min_alie",
            "multiwfn_max_lea",
            "qmdesc_fukui_plus",
            "qmdesc_fukui_minus",
            "qmdesc_fukui_dual",
            "qmdesc_partial_charge",
        ]

        # Add the xtb Fukui features in case they were calculated
        if calc_xtb_fukuis is True:
            xtb_fukui_features = [
                "xtb_fukui_plus",
                "xtb_fukui_minus",
                "xtb_fukui_dual",
                "xtb_fukui_zero",
                "xtb_partial_charge",
            ]
            self.target_features.extend(xtb_fukui_features)
    
    def get_values(self, feature_dict: dict, site_label: str) -> dict:
        """Method for getting the ortho, meta, or para features for all relevant C-H sites of the molecule."""
        self.feature_dict = feature_dict

        # Loop over the effective atom indices and collect the features for the ortho, meta, or para sites for all symmetry equivalent positions
        for site_idx, site_data in self.info_dict.items():
            if site_label == "ortho":
                neighbor_dict = site_data["ortho_atom_indices"]
            elif site_label == "meta":
                neighbor_dict = site_data["meta_atom_indices"]
            elif site_label == "para":
                neighbor_dict = site_data["para_atom_indices"]

            # Get the features for the ortho, meta, or para site for all symmetry equivalent positions
            for ipso_idx, neighbor_indices in neighbor_dict.items():
                try:
                    effective_features = self._get_neighbor_data(neighbor_indices)
                except Exception as e:
                    logging.error("Adduct-%s: error in OrthoMetaPara._get_values(): calculation of the ortho/meta/para features failed: %s: %s", site_idx, e.__class__.__name__, e)
                else:
                    # Save the effective features to the feature dictionary with an appropriate name
                    for idx, name in enumerate(self.target_features):
                        n = name.split("_", 1)
                        new_feature_name = f"{n[0]}_{site_label}_{n[1]}"
                        self.feature_dict[ipso_idx][new_feature_name] = float(effective_features[idx])

        return self.feature_dict
    
    def _get_neighbor_data(self, neighbor_indices: list) -> list:
        """Method for getting the mean feature vector for a list of sites."""
        # Zero padding in case the site is not defined (no para-site for 5-membered rings)
        if neighbor_indices == []:
            effective_features = [0.0 for _ in self.target_features]
            return effective_features
        
        # Get the features for all sites
        feature_matrix = []
        for idx in neighbor_indices:
            row_vector = self._get_feature_vector(idx)
            feature_matrix.append(row_vector)

        # Calculate the mean and return the effective feature vector
        effective_features = list(np.mean(feature_matrix, axis=0))
        return effective_features
    
    def _get_feature_vector(self, atom_idx: int) -> list:
        """Method for getting the relevant features from the feature dictionary for a given site."""
        atom_idx = str(atom_idx)
        features = self.feature_dict[atom_idx]
        feature_vector = [features[feature_name] for feature_name in self.target_features]
        return feature_vector
