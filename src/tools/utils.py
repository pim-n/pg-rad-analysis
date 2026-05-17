import os
import numpy as np


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EFF_FILES = {
    "HPGe": "relative_efficiency_HPGe.csv",
    "NaIR": "relative_efficiency_NaIR.csv",
    "NaIF": "relative_efficiency_NaIF.csv"
}

def return_efficiency_filename_path(det_type):
    match det_type:
        case "HPGe":
            efficiency_filename = EFF_FILES.get("HPGe")
            return os.path.join(BASE_DIR, "tools", efficiency_filename)
        case "NaIR":
            efficiency_filename = EFF_FILES.get("NaIR")
            return os.path.join(BASE_DIR, "tools", efficiency_filename)
        case "NaIF":
            efficiency_filename = EFF_FILES.get("NaIF")
            return os.path.join(BASE_DIR, "tools", efficiency_filename)
