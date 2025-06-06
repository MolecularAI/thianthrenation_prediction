File naming conventions
-----------------------
CONVSEARCH_*.out
    Log file of the conformer generation pipeline.

ENSEMBLEALL_*.xyz
    Conformational ensemble file with all conformers generated by RDKit.

ENSEMBLEXTBOPT_*.xyz
    Conformational ensemble with all the conformers that were optimized with xtb. For the PATTCH workflow, only one conformer gets optimized.

molden.input
    Molden input file for the multiWFN calculations of the substrate molecule.

MULTIWFNALIE_*.out
    Output file of multiWFN for the calculation of the average local ionization energy (ALIE).

MULTIWFNLEA_*.out
    Output file of multiWFN for the calculation of the electron attachment energy (LEA).

PATTCH_*.csv
    Features for each site of the molecule.

PATTCH_*.log
    Log file of the PATTCH workflow.

PATTCH_*.pkl
    Information dictionary used by PATTCH for the predictions. Contains for example atom indices information and also the individual RDKit mol objects.

XTBFUKUI_*.out
    Output file of xtb from the Fukui indices calculation (requested by adding -x to the PATTCH call).

XTBREFINED_*.out
    Output file of xtb from the final structure optimization.

XTBREFINED_*.xyz
    xyz coordinates file of the xtb-optimized structure.
