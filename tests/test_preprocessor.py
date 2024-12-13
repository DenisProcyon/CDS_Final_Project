import pytest
import pandas as pd
from datetime import datetime
from unittest.mock import patch, MagicMock

from preprocessors import (
    MetadataAvgTransformer,
    CountryAssigner,
    AgeAssigner
)

@pytest.fixture
def sample_metadata():
    md = pd.DataFrame({
        "MAKE": ["ford", "toyota", "chevrolet", "nissan", "honda"],
        "OVERALL_STARS": [4.5, 4.7, 4.2, 4.0, 4.6],
        "CURB_WEIGHT": [3500, 3200, 3300, 3100, 3000],
        "MIN_GROSS_WEIGHT": [4500, 4100, 4200, 4000, 3900]
    })
    return md

@pytest.fixture
def sample_data():
    data = pd.DataFrame({
        "Make": ["Ford", "Toyota", "Chevrolet", "Nissan", "Honda"],
        "Model": ["Silverado", "Civic", "Camry", "F-150", "Altima"],
        "Year": [2022, 2014, 2019, 2016, 2011],
        "Mileage": [18107, 13578, 63565, 46054, 134672],
        "Condition": ["Excellent", "Excellent", "Excellent", "Good", "Fair"]
    })
    return data

def test_metadata_avg_transformer(sample_data, sample_metadata, monkeypatch):
    with monkeypatch.context() as m:
        m.setattr("preprocessors.metadata", sample_metadata)
        columns = ["OVERALL_STARS", "CURB_WEIGHT", "MIN_GROSS_WEIGHT"]
        transformer = MetadataAvgTransformer(columns=columns)
        transformer.fit(sample_data)
        transformed = transformer.transform(sample_data)

        assert all(col in transformed.columns for col in columns), "No avg. values appeared"

        ford_data = transformed.loc[transformed["Make"].str.lower() == "ford"].iloc[0]
        assert ford_data["OVERALL_STARS"] == 4.5
        assert ford_data["CURB_WEIGHT"] == 3500
        assert ford_data["MIN_GROSS_WEIGHT"] == 4500

def test_country_assigner(sample_data):
    mock_mapper = {"ford": "USA", "toyota": "Japan", "chevrolet": "USA", "nissan": "Japan", "honda": "Japan"}

    def mock_init(self, brands, url):
        self.brands = brands

    def mock_getitem(self, item):
        return mock_mapper[item.lower()]

    with patch("country_mapper.CountryMapper.__init__", new=mock_init), \
         patch("country_mapper.CountryMapper.__getitem__", new=mock_getitem):
        transformer = CountryAssigner()
        transformer.fit(sample_data)
        transformed = transformer.transform(sample_data)

        assert "Country" in transformed.columns

        ford_country = transformed.loc[transformed["Make"] == "Ford", "Country"].iloc[0]
        assert ford_country == "USA"

def test_age_assigner(sample_data):
    transformer = AgeAssigner()
    transformer.fit(sample_data)
    transformed = transformer.transform(sample_data)

    assert "Age" in transformed.columns
    assert "Year" not in transformed.columns

    current_year = datetime.now().year
    for idx, row in sample_data.iterrows():
        expected_age = current_year - row["Year"]
        assert transformed.loc[idx, "Age"] == expected_age
