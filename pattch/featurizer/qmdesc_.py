"""Feature calculation with qmdesc."""

def get_qmdesc_values(
    handler: "qmdesc.handler.ReactivityDescriptorHandler",
    substrate_smiles: str,
    feature_dict: dict
    ) -> dict:
    """A function for calculating atom-centered features with the ML model qmdesc and for saving them into the feature dictionary.

    Parameters
    ----------
    handler : qmdesc.handler.ReactivityDescriptorHandler
        Handler object to make predictions with qmdesc.
    substrate_smiles : str
        SMILES string of the currently treated molecule.
    feature_dict : dict
        Feature dictionary with the atom indices as keys. The values are dictionaries with the feature name as keys and the numbers as values.
    
    Returns
    -------
    feature_dict : dict
        Feature dictionary updated with the new features.
    
    """
    # Make the qmdesc predictions
    qmdesc_results = handler.predict(substrate_smiles)

    # Write predicted values into the feature dictionary
    for atom_idx in feature_dict:
        feature_dict[atom_idx]["qmdesc_partial_charge"] = float(qmdesc_results["partial_charge"][int(atom_idx)])
        feature_dict[atom_idx]["qmdesc_fukui_plus"] = float(qmdesc_results["fukui_elec"][int(atom_idx)])
        feature_dict[atom_idx]["qmdesc_fukui_minus"] = float(qmdesc_results["fukui_neu"][int(atom_idx)])
        feature_dict[atom_idx]["qmdesc_fukui_dual"] = float(feature_dict[atom_idx]["qmdesc_fukui_plus"] - feature_dict[atom_idx]["qmdesc_fukui_minus"])
        feature_dict[atom_idx]["qmdesc_nmr"] = float(qmdesc_results["NMR"][int(atom_idx)])
    return feature_dict
