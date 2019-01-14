# -*- coding: utf-8 -*-
"""
Script to transform a list of cities to a list of pyowm locations

@author: Tâm Le Minh
"""
import numpy as np
import pandas as pd
import pyowm

##Retrieve api-key
with open('api-key/owm-api-key.txt', 'r') as key_file:
    key = key_file.read()

owm = pyowm.OWM(key)

##Summon Pyowm database
dic = owm.city_id_registry()

##Read cities list
cities = pd.read_csv("data/current-version/cities.csv", ';')

##Locations lookup in pyowm DB
locrows = []
for index, row in cities.iterrows():
    lookup = dic.locations_for(row['City'].decode("utf-8"), country=row['Country'])
    if lookup==[]:
        print("Cannot find: %s" % row['City'])
    elif len(lookup)>1 and not np.isnan(row['ID']):
        for lu in lookup:
            if lu.get_ID() == row['ID']:
                locrows.append([lu])
                break
    else:
        locrows.append([lookup[0]])
        
##Build location dataframe to export
id20s = []
elt_id20 = []
dfrows = locrows

for row in dfrows:
    cit = row[0]
    row.append(cit.get_ID())
    row.append(cit.get_lat())
    row.append(cit.get_lon())
    row[0] = cit.get_name().encode("utf-8")

df = pd.DataFrame(dfrows, columns=['City', 'ID', 'Lat', 'Lon'])

##Export to locations.csv file
df.to_csv('data/current-version/locations.csv')

print("Done")