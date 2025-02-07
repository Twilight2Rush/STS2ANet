import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point, LineString

def get_mids(json_path,id_path):
  # get all the manhattan locations
  gdf = gpd.read_file(json_path) # read taxi zone file
  m_zones = gdf['borough'].apply(lambda x:x=='Manhattan')
  m_remove103 = gdf['location_id'].apply(lambda x:x!='103')
  m_gdf = gdf[(m_zones) & (m_remove103)]
  m_ids = m_gdf['location_id']
  m_ids.to_csv(id_path)

def get_tripdata(data_path_list, m_ids, start_time, end_time):
  # get the trip data
  dfs = [] 
  for data_path in data_path_list:
    tripdata = pd.read_parquet(data_path)
    dfs.append(tripdata)
  whole_tripdata = pd.concat(dfs) # the whole tripdata
  # only keep the manhattan data
  m_zones_d = whole_tripdata['DOLocationID'].apply(lambda x:x in m_ids.values)
  m_zones_p = whole_tripdata['PULocationID'].apply(lambda x:x in m_ids.values)
  m_tripdata = whole_tripdata[(m_zones_d) & (m_zones_p)]
  m_tripdata = getDataByTime(m_tripdata, start_time, end_time)
  m_tripdata['count']=1
  m_tripdata['month'] = m_tripdata['tpep_pickup_datetime'].dt.month
  m_tripdata['date'] = m_tripdata['tpep_pickup_datetime'].dt.date
  m_tripdata['day'] = m_tripdata['tpep_pickup_datetime'].dt.dayofweek + 1
  m_tripdata['hour'] = m_tripdata['tpep_pickup_datetime'].dt.hour
  return m_tripdata

def getDataByTime(tripdata, start_time, end_time):
    tpep_pickup_datetime_requried = tripdata['tpep_pickup_datetime'].map(lambda x: x >= start_time )
    tpep_dropoff_datetime_requried = tripdata['tpep_dropoff_datetime'].map(lambda x: x <= end_time )
    filtered = tripdata[tpep_pickup_datetime_requried & tpep_dropoff_datetime_requried]
    filtered = filtered.reset_index()
    return filtered

def getDataByLoc(tripdata, PULocationID, DOLocationID):
    PULocationID_requried = tripdata['PULocationID'].map(lambda x : x==PULocationID)
    DOLocationID_requried = tripdata['DOLocationID'].map(lambda x : x==DOLocationID)
    filtered = tripdata[PULocationID_requried & DOLocationID_requried]
    filtered = filtered.reset_index()
    return filtered