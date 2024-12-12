import pandas as pd
from datetime import datetime

from pathlib import Path

from sklearn.base import BaseEstimator, TransformerMixin

from country_mapper import CountryMapper

metadata_path = Path(__file__).parent.parent.parent / "data/Safercar_data.csv"

metadata = pd.read_csv(metadata_path)

class MetadataAvgTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns
        self._brand_averages = None

    def fit(self, X, y=None):
        md = metadata.copy()
        md['MAKE'] = md['MAKE'].str.lower().str.strip()
        self._brand_averages = md.groupby('MAKE')[self.columns].mean()
        return self

    def transform(self, X):
        X = X.copy()
        X['Make'] = X['Make'].str.lower().str.strip()
        for column in self.columns:
            X[column] = X['Make'].map(self._brand_averages[column])
        return X

class CountryAssigner(BaseEstimator, TransformerMixin):
    def __init__(self, url="https://www.canstarblue.com.au/vehicles/car-country-of-origin/"):
        self.url = url
        self._mapper = None

    def fit(self, X, y=None):
        unique_brands = X["Make"].unique().tolist()
        self._mapper = CountryMapper(brands=unique_brands, url=self.url)
        return self

    def transform(self, X):
        X = X.copy()
        X['Country'] = X['Make'].apply(lambda make: self._mapper[make])
        return X

class AgeAssigner(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X["Age"] = datetime.now().year - X["Year"]
        X.drop(columns=["Year"], inplace=True)
        return X

class OutlierRemover(BaseEstimator, TransformerMixin):
    def __init__(self, column, threshold=3.0):
        self.column = column
        self.threshold = threshold
        self._lower_bound = None
        self._upper_bound = None

    def fit(self, X, y=None):
        q1 = X[self.column].quantile(0.25)
        q3 = X[self.column].quantile(0.75)
        iqr = q3 - q1
        self._lower_bound = q1 - self.threshold * iqr
        self._upper_bound = q3 + self.threshold * iqr
        return self

    def transform(self, X):
        X = X.copy()
        X = X[(X[self.column] >= self._lower_bound) & (X[self.column] <= self._upper_bound)]
        return X
