import argparse
from datetime import datetime
from .pattch import PATTCH
from .utils.check_dependencies import check_multiwfn_dependency


def main(
    input_file: str,
    n_procs: int,
    adduct_type: str,
    calc_xtb_fukuis: bool,
    ) -> None:
    """Main method that runs the PATTCH workflow as requested by the user.

    Parameters
    ----------
    input_file : str
        Path to the input .csv file for the PATTCH workflow. Required columns are 'substrate_smiles' and 'substrate_name'.
    n_procs : int
        Number of processors to use during the run. Defaults to 8.
    adduct_type : str
        Defines which adduct is used for probing the individual C-H positions. Must be one of ['tBu', 'Me', 'proton', 'TS']. Defaults to 'tBu'. 
        If something other than 'tBu' is used, only the features are calculated without running the classification model for predicting the reactivity.
    calc_xtb_fukuis : bool
        Defines whether or not Fukui coefficients and atomic partial charges are additionally calculated with xtb.
        They are not used as features for the final reactivity classification model.
    
    Returns
    -------
    None
    
    """
    p = PATTCH(
        n_procs=n_procs,
        adduct_type=adduct_type,
        calc_xtb_fukuis=calc_xtb_fukuis,
        )
    p.predict(input_file)


if __name__ == "__main__":
    # Command line inputs
    parser = argparse.ArgumentParser(
        prog="PATTCH",
        description="PATTCH is a hybrid SQM/ML workflow for predicting the reaction feasibility and site-selectivity of aromatic C-H thianthrenation reactions."
        )

    # Input file
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        required=True,
        help="Path to the input .csv file for the PATTCH workflow. Required columns are 'substrate_smiles' and 'substrate_name'.",
    )

    # Number of processors
    parser.add_argument(
        "-p",
        "--processors",
        type=int,
        default=8,
        help="Number of processors to use during the run. Defaults to 8.",
    )

    # Which type of adduct should be used
    parser.add_argument(
        "-a",
        "--adduct_type",
        type=str,
        default="tBu",
        choices=["tBu", "Me", "proton", "TS"],
        help="Defines which adduct is used for probing the individual C-H positions. Must be one of ['tBu', 'Me', 'proton', 'TS']. Defaults to 'tBu'. If something other than 'tBu' is used, only the features are calculated without running the classification model for predicting the reactivity.",
    )

    # Decide if xtb fukui coefficients should be calculated
    parser.add_argument(
        "-x",
        "--xtb_charges_fukuis",
        action=argparse.BooleanOptionalAction,
        type=bool,
        help="Add '-x' or '--xtb_charges_fukuis' to the PATTCH call if the atomic partial charges and Fukui coefficients from xtb should also be calculated. They are not used in the classification model of the PATTCH workflow.",
    )

    args = parser.parse_args()

    print("===================================================")
    print(" Predictive model for the Aromatic ThianThrenation ")
    print("    reaction for C-H functionalization (PATTCH)    ")
    print("===================================================")
    start_time = datetime.now()
    print(f"Start time: {start_time}")
    print()

    # Check for Multiwfn dependency
    check = check_multiwfn_dependency()
    if check is not None:
        print(check)
    else:
        # Call main function
        main(
            input_file=args.input,
            n_procs=args.processors,
            adduct_type=args.adduct_type,
            calc_xtb_fukuis=args.xtb_charges_fukuis,
        )

    end_time = datetime.now()
    run_time = end_time - start_time
    print()
    print(f"End time: {end_time}")
    print(f"Total run time: {round(run_time.total_seconds(), 1)} seconds.")
    print("===================================================")
    print("                PATTCH run finished                ")
    print("===================================================")
