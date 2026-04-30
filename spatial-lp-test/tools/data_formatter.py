import csv
import pandas as pd


class DataFormatter:

    @staticmethod
    def __csv_to_dict(filename):
        """
        Transform the data from csv file to dictionary with headers as keys.
        :return: dictionary with headers as keys or ValueError
        """

        with open(filename, newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            headers = [header.strip() for header in next(reader)]
            data_dict = {header: [] for header in headers}
            for row in reader:
                for i, value in enumerate(row):
                    try:
                        data_dict[headers[i]].append(float(value))
                    except ValueError as err:
                        return err
        return data_dict

    def get_dataframe(self, filename):
        data = self.__csv_to_dict(filename)
        return pd.DataFrame(data=data)


formatter = DataFormatter()
