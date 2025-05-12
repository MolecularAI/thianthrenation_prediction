"""Graphical visualization of the prediction results as individual and grid image(s) of molecules."""

import os
import math
from rdkit import Chem
from rdkit.Chem import Draw
from PIL import Image
from io import BytesIO


def generate_image(
    mol: Chem.rdchem.Mol,
    name: str,
    true_indices: list,
    false_indices: list,
    ) -> Image.Image:
    """A function for generating a molecular structure image representation with highlighted atoms with RDKit.
    
    Parameters
    ----------
    mol : Chem.rdchem.Mol
        RDKit mol object of the molecule that will be plotted.
    name : str
        Name of the molecule that will be plotted as specified by the user.
    true_indices : list
        Atom indices of the carbon atoms of the C-H bonds that were predicted reactive.
    false_indices : list
        Atom indices of the carbon atoms of the C-H bonds that were predicted unreactive.
    
    Returns
    -------
    image : Image.Image
        Generated image.

    """
    # Color and radius settings
    color_true = (255/255, 193/255, 7/255)
    color_false = (216/255, 27/255, 96/255)
    r = 0.5

    # Set up dictionaries for atom highlighting
    highlights = {}
    radii = {}
    for idx in true_indices:
        highlights[idx] = [color_true]
        radii[idx] = r

    for idx in false_indices:
        highlights[idx] = [color_false]
        radii[idx] = r

    # Drawing options
    drawing = Draw.MolDraw2DCairo(400, 400)
    options = drawing.drawOptions()
    options.fixedBondLength = 20
    options.legendFontSize = 20
    options.explicitMethyl = False
    options.addAtomIndices = True
    options.annotationFontScale = 1.0
    options.setAnnotationColour((0/255, 0/255, 0/255, 140/255))
    options.useBWAtomPalette()

    # Draw and return image
    drawing.DrawMoleculeWithHighlights(mol, name, highlights, {}, radii, {})
    drawing.FinishDrawing()
    image = Image.open(BytesIO(drawing.GetDrawingText()))
    return image


def make_grid_image(image_list: list, n_col: int = 5) -> None:
    """A function for generating a grid image from a list of images.
    
    Parameters
    ----------
    image_list : list
        A list of all the images that should be included in the final grid image.
    n_col : int
        Number of columns the grid image should have.
    
    Returns
    -------
    None
    
    """
    # Determine dimensions
    shape = (math.ceil(len(image_list) / n_col), n_col)
    width = image_list[0].width
    height = image_list[0].height

    # Initialize the grid image
    image_size = (width * shape[1], height * shape[0])
    image = Image.new("RGBA", image_size)

    # Populate the grid image with the individual images
    for row in range(shape[0]):
        for col in range(shape[1]):
            offset = (width * col, height * row)
            idx = row * shape[1] + col
            if idx < len(image_list):
                image.paste(image_list[idx], offset)

    # Save the grid image
    image.save(os.path.join("prediction_results", "_all.png"))


def make_single_image(
    results_df: "pd.DataFrame",
    info_dict: dict,
    substrate_name: str
    ) -> Image.Image:
    """A function for visualizing the prediction results with a color-coded molecular structure.

    Parameters
    ----------
    results_df : pd.DataFrame
        Dataframe with the prediction results.
    info_dict : dict
        Dictionary with general information on the molecule (e.g., mol object, SMILES, potential sites of substitution).
    substrate_name : str
        Name of the compound as specified by the user.
    
    Returns
    -------
    image : Image.Image
        Generated image.
    """
    # Separate between sites predicted reactive and unreactive
    true_indices = []
    false_indices = []
    for _, row_data in results_df.iterrows():
        idx = row_data["_site_idx"]
        all_indices = info_dict["site_info"][idx]["site_indices"]
        if row_data["is_predicted_reactive"] is True:
            true_indices.extend(all_indices)
        else:
            false_indices.extend(all_indices)

    # New mol object for formatting reasons (to keep the conformers in the actual mol object)
    mol = Chem.Mol(info_dict["substrate_info"]["mol"])
    mol = Chem.RemoveHs(mol)
    mol.RemoveAllConformers()

    # Generate the image
    image = generate_image(
        mol=mol,
        name=substrate_name,
        true_indices=true_indices,
        false_indices=false_indices,
    )

    # Save and return the image
    image.save(os.path.join("prediction_results", f"{substrate_name}.png"))
    return image
