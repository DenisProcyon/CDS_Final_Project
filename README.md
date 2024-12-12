# CDS_Final_Project

## Preprocessors Module

Since our pipeine is somewhat custom, this module provides a set of custom **scikit-learn-compatible** transformers designed to prepare and enrich vehicle-related data before it reaches a predictive model. Each transformer is a subclass of `BaseEstimator` and `TransformerMixin`, ensuring seamless integration into sklearn `Pipeline` and `GridSearchCV` workflows.

### Overview of Transformers

1. **MetadataAvgTransformer**
   - **Purpose:** Enriches the dataset with average metadata values (e.g., safety ratings, weights) based on the car's make.
   - **Process:**  
     - Uses a global `metadata` DataFrame containing aggregated brand information.
     - Matches each entry’s `Make` to the averaged columns (`OVERALL_STARS`, `CURB_WEIGHT`, `MIN_GROSS_WEIGHT`) and updates the DataFrame accordingly.

2. **CountryAssigner**
   - **Purpose:** Determines the country of origin for each car brand.
   - **Process:**  
     - Utilizes `CountryMapper` to map `Make` to a corresponding country.
     - Adds a new column `Country` to the dataset.

3. **AgeAssigner**
   - **Purpose:** Computes the vehicle’s age from its manufacturing year.
   - **Process:**  
     - Calculates `Age = current_year - Year`.
     - Drops the `Year` column.

4. **OutlierRemover**
   - **Purpose:** Filters out outliers from a specified numerical column to improve model robustness.
   - **Process:**  
     - Calculates Interquartile Range (IQR) and determines upper and lower bounds.
     - Retains only the rows whose values fall within these bounds.

### Example Usage

```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeRegressor

numeric_features = ["Mileage", "OVERALL_STARS", "CURB_WEIGHT", "MIN_GROSS_WEIGHT", "Age"]
categorical_features = ["Condition", "Country", "Make"]

preprocessing = ColumnTransformer([
    ("ohe", OneHotEncoder(handle_unknown='ignore'), categorical_features),
    ("num", "passthrough", numeric_features)
])

pipeline = Pipeline([
    ("metadata_avg", MetadataAvgTransformer(columns=["OVERALL_STARS", "CURB_WEIGHT", "MIN_GROSS_WEIGHT"])),
    ("country", CountryAssigner()),
    ("age", AgeAssigner()),
    ("outlier_remover", OutlierRemover(column="Mileage", threshold=3)),
    ("preprocessing", preprocessing),
    ("regressor", DecisionTreeRegressor(random_state=42))
])

pipeline.fit(X_train, y_train)
predictions = pipeline.predict(X_test)
```


## DataPlotter Module

The `DataPlotter` class provides a set of visualization methods for quick exploratory data analysis of both categorical and numerical features. Leveraging **pandas**, **matplotlib**, and **seaborn**, it allows you to easily plot bar charts, distributions of numerical columns, and distributions of categorical columns. It comes with sensible default configurations for styling and layout.


### Methods Overview

1. **`plot_barchart(data: pd.Series, title: str, xlabel: str, ylabel: str, figsize: tuple = None)`**
   - **Description:**  
     Creates a bar chart of a given `pd.Series` with customizable title and axis labels.
   - **Usage Example:**
     ```python
     plotter = DataPlotter()
     counts = df['Make'].value_counts()
     plotter.plot_barchart(counts, title="Car Make Distribution", xlabel="Make", ylabel="Count")
     ```

2. **`plot_numerical_distribution(data: pd.DataFrame, columns: list[str])`**
   - **Description:**  
     Plots the distribution (via `seaborn.kdeplot`) of specified numerical columns, helping visualize their spread and shape.
   - **Usage Example:**
     ```python
     plotter = DataPlotter()
     numeric_cols = ['Mileage', 'Price']
     plotter.plot_numerical_distribution(df, numeric_cols)
     ```

3. **`plot_all_categorical(data: pd.DataFrame)`**
   - **Description:**  
     Automatically detects and plots the distribution of all categorical columns in the `DataFrame`. Each categorical column is represented by a bar chart of its value counts.
   - **Usage Example:**
     ```python
     plotter = DataPlotter()
     plotter.plot_all_categorical(df)
     ```

### Customization

- **Default Configurations:**  
  The `DataPlotter` initializes with a set of default configurations:
  - `figsize`: (12, 8)
  - `color`: "skyblue"
  - `title_fontsize`: 16
  - `label_fontsize`: 14
  - `tick_labelsize`: 12
  
  You can override these defaults by passing `figsize` in the plotting methods or modifying the class to accept more parameters.

### Example

```python
import pandas as pd

df = pd.DataFrame({
    'Make': ['Ford', 'Toyota', 'Ford', 'Chevrolet', 'Toyota', 'Ford'],
    'Condition': ['Good', 'Excellent', 'Fair', 'Excellent', 'Good', 'Good'],
    'Mileage': [20000, 30000, 15000, 40000, 25000, 22000],
    'Price': [18000, 22000, 16000, 25000, 21000, 19000]
})

plotter = DataPlotter()

plotter.plot_all_categorical(df)

plotter.plot_numerical_distribution(df, ['Mileage', 'Price'])

condition_counts = df['Condition'].value_counts()
plotter.plot_barchart(condition_counts, title="Condition Distribution", xlabel="Condition", ylabel="Count")
```

## CountryMapper Module

The `CountryMapper` class provides a  way to determine the country of origin for various car brands. Given a list of brands and a URL containing a reference table, it scrapes the webpage and constructs a mapping from brand names to countries. With this mapper, you can quickly enrich your dataset by adding a `Country` column based on each car’s `Make`.

## Testing with Pytest

### Key Testing Strategies Demonstrated

- **Isolation:**  
  Tests focus on one piece of functionality at a time. Each transformer is tested independently to confirm it performs its role correctly.
  
- **Repeatability:**  
  The tests can be run multiple times with the same results, ensuring deterministic outputs. Mocking ensures no external dependencies affect results.

- **Readability and Maintainability:**
  By using descriptive test function names and in-code comments, it’s easier for other developers to understand what’s being tested and why.

### Example Test Explanation

In `test_metadata_avg_transformer`, we:
1. Monkeypatch the global `metadata` variable so the transformer uses the `sample_metadata` fixture.
2. Instantiate `MetadataAvgTransformer` and fit-transform the sample data.
3. Check that new columns (`OVERALL_STARS`, `CURB_WEIGHT`, `MIN_GROSS_WEIGHT`) appear.
4. Verify that the values for `Ford` match those specified in the mock metadata.

Similarly, in `test_country_assigner`, we:
1. Mock `CountryMapper` methods to return predefined countries for each brand.
2. Fit-transform using `CountryAssigner`.
3. Assert that the `Country` column is created and matches expected values.

Lastly, `test_age_assigner`:
1. Checks that `Age` is computed and `Year` is removed.
2. Ensures that the computed `Age` equals `current_year - Year`.

## Running the Tests

Assuming you have pytest installed, you can run:
```bash
cd tests
pytest test_preprocessor.py  
```

## FastAPI Endpoint for Car Price Prediction

- **Model Loading:**  
  On startup, the application loads a pre-trained pipeline from `best_model.pkl`. 

- **Data Validation with Pydantic:**  
  The `CarFeatures` Pydantic model defines the expected input schema for each request.

- **Predict Endpoint (`/predict`):**  
  Sends a JSON object containing a single car's features. The API converts this to a `pandas.DataFrame` and passes it through the pipeline to generate a price prediction. The response is a JSON object with a single `prediction` key.

  **Request Example:**
  ```json
  {
    "Make": "Ford",
    "Model": "F-150",
    "Year": 2020,
    "Mileage": 30000,
    "Condition": "Excellent"
  }

