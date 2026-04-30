import asyncio
import numpy as np
import pandas as pd
from pathlib import Path


class Extractor:
    PARAMS_NAMES = [
        "MEAN_BKG_CPS",
        "BKG_STD",
        "X_POS",
        "Y_POS",
        "MAX_X",
        "MAX_Y",
        "MAX_BKG",
        "MAX_CPS",
        "CPS",
        "LOWER_X",
        "UPPER_X",
        "LOWER_Y",
        "UPPER_Y"
    ]

    def __init__(self, directory, offset=0):
        self.directory = directory
        self.offset = offset
            
    async def __read_csv_files(self, filename):
        df = pd.read_csv(filename)
        max_ind = np.argmax(df.iloc[:,2]) + self.offset
        return {
                "MEAN_BKG_CPS": [df.iloc[:,3].mean()],
                "BKG_STD": [df.iloc[:,3].std()],
                "X_POS": [df.iloc[:,0].values],
                "Y_POS": [df.iloc[:,1].values],
                "MAX_X": [df.iloc[max_ind, 0]],
                "MAX_Y": [df.iloc[max_ind, 1]],
                "MAX_BKG": [df.iloc[max_ind, 3]],
                "MAX_CPS": [df.iloc[max_ind, 2]],
                "CPS": [df.iloc[:,2].values],
                "LOWER_X": [df.iloc[:,0].min()],
                "UPPER_X": [df.iloc[:,0].max()],
                "LOWER_Y": [df.iloc[:,1].min()],
                "UPPER_Y": [df.iloc[:,1].max()]
        }

    async def __process_files(self, csv_files, params):
        tasks = [self.__read_csv_files(csv_file) for csv_file in csv_files]
        results = await asyncio.gather(*tasks)

        combined_data = {param: [] for param in params}

        for result in results:
            for param in params:
                combined_data[param].extend(result[param])

        return combined_data

    async def extract(self):
        csv_files = list(self.directory.glob("*.csv"))
        combined_data = await self.__process_files(csv_files, self.PARAMS_NAMES)
        return combined_data
