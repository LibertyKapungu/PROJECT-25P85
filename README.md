project-repo/
├── .gitignore             # Tells Git which files to ignore (e.g., data, secrets)
├── README.md              # Main project documentation (IMPORTANT!)
├── requirements.txt       # List of Python packages needed to run the project
|
├── data/                  # All project data
│   ├── raw/               # Original, unmodified data
│   └── processed/         # Cleaned, transformed, or feature-engineered data
|
├── notebooks/             # Jupyter notebooks for exploration and analysis
│   ├── 01_data_exploration.ipynb
│   └── 02_model_prototyping.ipynb
|
├── src/                   # Source code for your project
│   ├── __init__.py
│   ├── data_processing.py # Functions for cleaning and preparing data
│   ├── modeling.py        # Functions for training and evaluating models
│   └── visualization.py   # Functions for creating plots
|
└── results/               # All generated output (this is where you'd put shared metrics)
    ├── figures/           # Generated plots and charts
    ├── metrics/           # CSVs or text files with model scores, etc.
    └── models/            # Saved, trained models (.pkl, .h5, etc.)

