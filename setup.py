from setuptools import setup, find_packages

setup(
    name="CDS Final Project",  
    version="0.1.0",
    description="Decision Tree Based Car Price Prediction",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="Denis, Sasha, Masha, Simon",
    author_email="denis.shadrin@bse.eu",
    license="",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    include_package_data=True,
    install_requires=[
        "requests",
        "beautifulsoup4",
        "pandas",
        "numpy",
        "matplotlib",
        "seaborn",
        "scikit-learn",
        "joblib",
        "fastapi",
        "uvicorn"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.12",
)
