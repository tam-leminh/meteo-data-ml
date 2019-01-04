# meteo-data-regressor

## Setup

- Get an Open Weather Map [API key](https://openweathermap.org/api)
- Copy the key in an empty text file owm-api-key.txt, this file should be placed in the folder api-key
- Install [pyowm](https://github.com/csparpa/pyowm)
- Execute Cities2Locations.py (or run the notebook Cities2Locations.ipynb) to create locations.csv in data/current-version

## Data acquisition

- Execute DownloadData.py (or run the notebook DownloadData.ipynb)
- If successful, the data is stored in data/current-version

## Analysis

- Run the notebook Meteo.ipynb
- Replace the csv to read name with your own datafile name