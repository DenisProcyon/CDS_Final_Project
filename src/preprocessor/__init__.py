import pandas as pd
import numpy as np

from datetime import datetime

from country_mapper import CountryMapper

class Preprocessor:
    def __init__(self, data: pd.DataFrame, metadata: pd.DataFrame):
        self.data = data
        self.metadata = metadata

    def assign_metadata_avg(self, columns: list[str]) -> pd.DataFrame:
        self.metadata['MAKE'] = self.metadata['MAKE'].str.lower().str.strip()
        self.data['Make'] = self.data['Make'].str.lower().str.strip()

        brand_averages = self.metadata.groupby('MAKE')[columns].mean()

        for column in columns:
            self.data[column] = self.data['Make'].map(brand_averages[column])

        return self.data

    def assign_countries(self) -> pd.DataFrame:
        mapper = CountryMapper(
            brands=self.data["Make"].unique().tolist(),
            url="https://www.canstarblue.com.au/vehicles/car-country-of-origin/"
        )

        self.data['Country'] = self.data['Make'].apply(lambda make: mapper[make])

        return self.data
    
    def assign_age(self) -> pd.DataFrame:
        self.data["Age"] = datetime.now().year - self.data["Year"]
        self.data = self.data.drop(columns=["Year"])

        return self.data

    def remove_outliers(self, column: str, threshold: float = 3.0) -> pd.DataFrame:
        q1 = self.data[column].quantile(0.25)
        q3 = self.data[column].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - threshold * iqr
        upper_bound = q3 + threshold * iqr
        self.data = self.data[(self.data[column] >= lower_bound) & (self.data[column] <= upper_bound)]

        return self.data

    def transform_cat_to_num(self, columns: list[str]) -> pd.DataFrame:
        for column in columns:
            dummies = pd.get_dummies(self.data[column], prefix=column)
            self.data = pd.concat([self.data, dummies], axis=1)
            self.data.drop(column, axis=1, inplace=True)

        return self.data
