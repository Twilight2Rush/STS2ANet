import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point, LineString
import matplotlib.pyplot as plt

def get_poi(json_path):
  poi_gdf = gpd.read_file(json_path)
  m_poi = poi_gdf['borough'].apply(lambda x:x=='1')
  m_poi_gdf = poi_gdf[m_poi]
  return m_poi_gdf

def get_zones(json_path):
  gdf = gpd.read_file(json_path) # read taxi zone file
  m_zones = gdf['borough'].apply(lambda x:x=='Manhattan')
  m_remove103 = gdf['location_id'].apply(lambda x:x!='103')
  m_gdf = gdf[(m_zones) & (m_remove103)]
  m_gdf.set_index('objectid',inplace=True)
  return m_gdf

def getZone(point,zone_gdf):
  for index,zone in zone_gdf.iterrows():
    if point.within(zone['geometry']):
      return zone['location_id']
  return 1000 # not in any zone

def poi_vec(loc_id,poi_gdf):
  poi_list = np.zeros(13)
  loc_gdf = poi_gdf[poi_gdf['zone_id'].apply(lambda x:x==loc_id)]
  value_counts = loc_gdf['facility_t'].value_counts(normalize=True)
  for i, v in value_counts.items():
    poi_list[int(i)-1] = v
  return poi_list

def main():
    json_path = '../data/Points Of Interest.geojson' # poi geojson path
    zone_json_path = '../data/NYC Taxi Zones.geojson' # zone geojson path
    # load to geopandas dataframe
    poi_gdf = get_poi(json_path)  
    zone_gdf = get_zones(zone_json_path)
    poi_gdf['zone_id'] = poi_gdf['geometry'].apply(getZone, zone_gdf=zone_gdf) # apply poi points into zones
    zone_gdf['poi_vec'] = zone_gdf['location_id'].apply(poi_vec, poi_gdf=poi_gdf) # final poi vector
    print(zone_gdf['poi_vec'])
    return zone_gdf['poi_vec']

main()