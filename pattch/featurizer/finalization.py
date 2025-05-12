"""Finalization of the features before model inference, e.g. adding molecule-level features or introducing steric penalties."""

import operator
import numpy as np


def get_effective_features(feature_dict: dict, info_dict: dict) -> dict:
    """A function for calculating the average feature vector for one or more symmetry equivalent positions.

    Parameters
    ----------
    feature_dict : dict
        Feature dictionary with the atom indices as keys. The values are dictionaries with the feature name as keys and the numbers as values.
    info_dict : dict
        Dictionary with general information on the molecule (e.g., mol object, SMILES, potential sites of substitution).

    Returns
    -------
    feature_dict : dict
        Feature dictionary updated with the new features.
    
    """
    # Loop over the effective atom indices
    for effective_atom_idx, site_data in info_dict["site_info"].items():
        feature_matrix = []

        # Loop over all atom indices that belong to a given effective atom index
        for atom_idx in site_data["site_indices"]:
            row = []

            # Get the individual features
            for feature in feature_dict[str(atom_idx)].values():
                row.append(feature)
            feature_matrix.append(row)

        # Calculate the average of the features
        effective_features = list(np.mean(feature_matrix, axis=0))

        # Update the feature dictionary
        for idx, feature_name in enumerate(feature_dict[effective_atom_idx]):
            feature_dict[effective_atom_idx][feature_name] = float(effective_features[idx])

    # Remove everything except for the relevant atom indices
    feature_dict = {atom_idx: features for atom_idx, features in feature_dict.items() if atom_idx in info_dict["site_info"]}

    return feature_dict


def adjust_features(feature_dict: dict, adduct_type: str, calc_xtb_fukuis: bool) -> dict:
    """A function that introduces the steric penalty term to certain features.

    Parameters
    ----------
    feature_dict : dict
        Feature dictionary with the atom indices as keys. The values are dictionaries with the feature name as keys and the numbers as values.
    adduct_type : str
        Name of the chemical agent that is used to probe a given C-H group. "tBu" for PATTCH.
    calc_xtb_fukuis : bool
        Defines whether xtb Fukui coefficients have been calculated or not.

    Returns
    -------
    feature_dict : dict
        Feature dictionary updated with the new features.

    """
    for features in feature_dict.values():
        # Treat xtb Fukui value separately depending on whether it was calculated or not
        if calc_xtb_fukuis is True:
            features["_xtb_fukui_minus_raw"] = features["xtb_fukui_minus"]  # Save the initial values
            features["xtb_fukui_minus"] = features["xtb_fukui_minus"] - 0.183673469387755 * features["morfeus_ortho_buried_volume"]  # Adjust feature

        # Save the initial values
        features["_qmdesc_fukui_minus_raw"] = features["qmdesc_fukui_minus"]
        features["_multiwfn_min_alie_raw"] = features["multiwfn_min_alie"]
        features["_xtb_position_energy_raw"] = features["xtb_position_energy"]
        
        # Adjust the features
        features["qmdesc_fukui_minus"] = features["qmdesc_fukui_minus"] - 0.132653061224489 * features["morfeus_buried_volume"]
        features["multiwfn_min_alie"] = features["multiwfn_min_alie"] + 1.63265306122448 * features["morfeus_ortho_buried_volume"]
        
        if adduct_type == "proton":
            features["xtb_position_energy"] = features["xtb_position_energy"] - 285.714285714285 * features["morfeus_ortho_buried_volume"]
        if adduct_type == "Me":
            features["xtb_position_energy"] = features["xtb_position_energy"] - 224.489795918367 * features["morfeus_ortho_buried_volume"]
        if adduct_type == "tBu":
            features["xtb_position_energy"] = features["xtb_position_energy"] - 244.897959183673 * features["morfeus_ortho_buried_volume"]
        if adduct_type == "TS":
            features["xtb_position_energy"] = features["xtb_position_energy"] + 367.34693877551 * features["morfeus_ortho_buried_volume"]

    return feature_dict


def add_categorical_feature(feature_dict: dict, feature_name: str, target: str) -> dict:
    """A functions that adds the molecule-level categorical features to the feature dictionary.

    Parameters
    ----------
    feature_dict : dict
        Feature dictionary with the atom indices as keys. The values are dictionaries with the feature name as keys and the numbers as values.
    feature_name : str
        Feature that is used to compare all sites of the molecule to get the molecule-level descriptor.
    target : str
        Whether the highest or the lowest value of the feature is defining the categorical feature value.

    Returns
    -------
    feature_dict : dict
        Feature dictionary updated with the new features.
    
    """
    # Find largest or smallest value
    if target == "highest":
        extreme_value = -np.inf
        comparison_operator = operator.gt
    if target == "lowest":
        extreme_value = np.inf
        comparison_operator = operator.lt
    extreme_atom_idx = None

    for atom_idx, features in feature_dict.items():
        if comparison_operator(features[feature_name], extreme_value):
            extreme_value = features[feature_name]
            extreme_atom_idx = atom_idx

    # Update feature dict for all positions
    for atom_idx in feature_dict:
        cat_val = 0
        if atom_idx == extreme_atom_idx:
            cat_val = 1
        feature_dict[atom_idx][f"categorical_{target}-{feature_name}"] = cat_val

    return feature_dict
