gsod
==============================

Analysis of historical weather data from NOAAs Global Summary of the Day (GSOD).

Project Organization
------------

    ├── LICENSE
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── interim        <- Intermediate data used for processing, such as manually created data.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump from BigQuery.
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── sql_queries        <- Queries that were manually run against the GSOD dataset in BigQuery
    |                         to produce the data files in `data/raw/`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   |
    │   ├── analysis       <- Code to analyze raw data
    │   |   ├── continuity.py   <- Module to analyze station continuity.
    │   │   └── precipitation.py    <- Modeule to analyze precipitation data.
    │   |
    │   ├── data           <- Scripts to download or generate data
    │   |   ├── loaders.py <- Module to load and process raw data.
    │   │   └── make_dataset.py <- The final, canonical data sets.
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py    <- (empty)
    │
    ├── environment_direct.yml   <- The environment file for reproducing the direct dependencies of the
    │                                analysis environment, generated with `conda env export --from-history`
    │
    └── environment_strict.yml   <- The environment file for reproducing the ALL dependencies of the
                                     analysis environment, generated with `conda env export`


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
