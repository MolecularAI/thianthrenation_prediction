"""Functions for checking if required dependencies are available."""

import os
import shutil
from typing import Union


def check_multiwfn_dependency() -> Union[None, str]:
    """A function that checks for the required Multiwfn dependency.

    Parameters
    ----------
    None

    Returns
    -------
    check_result : None or str
        Returns None if every check was passed and the reason for failure as a string otherwise.
    
    """
    check_result = None
    
    try:
        w_result = shutil.which("Multiwfn_noGUI", mode=os.F_OK | os.X_OK)
    except Exception:
        check_result = "ERROR: Multiwfn path was not correctly added to the PATH variable. Ensure that the 'Multiwfn_noGUI' is executable."
    else:
        if w_result is None:
            check_result = "ERROR: Multiwfn was not added to the PATH variable."
    
    return check_result
