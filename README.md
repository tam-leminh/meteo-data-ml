# meteo-data-regressor

## Setup

- Get an Open Weather Map [API key](https://openweathermap.org/api)
- Copy the key in an empty text file owm-api-key.txt, this file should be placed in the folder api-key
- Install [pyowm](https://github.com/csparpa/pyowm)
- Install [py-earth](https://github.com/scikit-learn-contrib/py-earth)
- Install [Basemap](https://matplotlib.org/basemap)
- Execute Cities2Locations.py to create locations.csv in data/current-version

## Data acquisition

- Execute DownloadData.py
- If successful, the data is stored in data/current-version

## Analysis

- Use the example scripts or notebooks
- Replace the csv to read name with your own datafile name