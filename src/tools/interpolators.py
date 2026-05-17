import pandas as pd
import pytensor.tensor as pt
from scipy.interpolate import interp1d
from tools.base_interplolator import BaseInterpolator


class Interpolator(BaseInterpolator):

    def __init__(self, data: pd.DataFrame):
        self.data = data

    def interpolate(self, x: pd.Series) -> pd.Series:
        pass


class AttenuationInterpolator(Interpolator):

    def interpolate(self, *args, scaling_factor=1000) -> float:
        x = self.data["energy_mev"].to_numpy()
        y = self.data["mu"].to_numpy()
        f = interp1d(x, y)
        return f(args[0] / scaling_factor)


class EfficiencyInterpolator(Interpolator):

    def interpolate(self, arg) -> float:
        x = self.data["Angle"].values
        y = self.data["E_rel_662_90"].values
        f = interp1d(x, y, fill_value=True)
        return f(arg)
