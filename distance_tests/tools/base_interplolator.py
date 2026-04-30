from abc import ABC, abstractmethod


class BaseInterpolator(ABC):

    @abstractmethod
    def interpolate(self, dataframe):
        pass
