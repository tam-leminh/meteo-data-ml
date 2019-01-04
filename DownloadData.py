import numpy as np
import pandas as pd
import pyowm
import time
import datetime

def owm_query(id_list, temp, it = 0):
    try:
        it = it + 1
        result = owm.weather_at_ids(id_list)
        temp.append(result)
    except:
        print("it = %i" % it)
        if it < 10:
            time.sleep(1)
            owm_query(id_list, temp, it)
        else:
            print("Query failed")

if __name__=='__main__':
            
    #Retrieve api-key
    with open('api-key/owm-api-key.txt', 'r') as key_file:
        key = key_file.read()
    
    owm = pyowm.OWM(key)
    
    #Read cities list
    locations = pd.read_csv("data/current-version/locations.csv", ',')
    locations = locations.drop(['Unnamed: 0'], axis=1)
    
    observation_list = []
    id20s = []
    elt_id20 = []
    c = 0
    
    #Regroup IDs in lists of 20, maximum observation number per query
    for c in range(0, len(locations['ID'])):
        elt_id20.append(int(locations['ID'][c]))
        if len(elt_id20) == 20 or c == len(locations['ID'])-1:
            id20s.append(elt_id20)
            elt_id20 = []
        c = c + 1
    
    #OWM queries to get observations
    for id20 in id20s:
        temp = []
        owm_query(id20, temp)
        if temp == []:
            print("Warning: Missing data due to failed queries")
        else:
            observation_list.append(temp[0])
        time.sleep(1)
        
    #Get weather and temperature for each city
    weathers = []
    temps = []
    
    for obss in observation_list:
        for obs in obss:
            weathers.append(obs.get_weather())
            
    for weather in weathers:
        temps.append(weather.get_temperature(unit='celsius'))
        
    dfrows = []
    for c in range(0, locations.shape[0]):
        dfrows.append(temps[c]['temp'])
        
    df = pd.DataFrame({ 'City' : locations['City'].tolist(),
                       'ID' : locations['ID'].tolist(), 
                       'Lat' : locations['Lat'].tolist(), 
                       'Lon' : locations['Lon'].tolist(), 
                       'Temp' : dfrows })
    
    x = datetime.datetime.now()
    df.to_csv('data/current-version/Temp-{:%Y_%m_%d-%H_%M}.csv'.format(x))