#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@filename: utils.py
@description: some functions for data processing   
@time: 2023/08/23
@author: Haoli WANG
@Version: 1.0
'''

import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point, LineString, Polygon, MultiPolygon
import matplotlib.pyplot as plt
from fiona.crs import from_epsg
from scipy.spatial import distance_matrix
np.set_printoptions(suppress=True)


def read_zones(json_path,zone_json_path):
  """
  Description: 
    used to read the Manhattan taxi zones
  Parameters:
    json_path --- the input geojson file path
  Outputs:
    a geojson file --- which includes the processed manhattan zones info
  """
  gdf = gpd.read_file(json_path) # read taxi zone file
  m_zones = gdf['borough'].apply(lambda x:x=='Manhattan') # Manhattan zones
  m_remove103 = gdf['location_id'].apply(lambda x:x!='103') # remove 103
  m_gdf = gdf[(m_zones) & (m_remove103)]
  m_gdf['objectid'] = m_gdf['objectid'].astype('int')
  m_gdf = m_gdf.sort_values('objectid')
  m_gdf.rename(columns={'objectid':'id'},inplace = True)
  m_gdf.set_index('id',inplace=True)
  m_gdf.to_file(zone_json_path, driver='GeoJSON')
  
def load_zones(zone_json_path):
  """
  Description: 
    used to read the zones csv file and output a geo dataframe of Manhattan zones
  Parameters:
    zone_json_path --- the input zone geojson path
  Returns:
    gdf --- a geodatframe
  """
  gdf = gpd.read_file(zone_json_path)
  gdf.set_index('id',inplace=True)
  return gdf

def compare_and_assign(value, threshold):
  """
  Description: 
    used to compare two zones' distance with a threshold
  Parameters:
    value --- the distance of two zones
  Returns:
    0 or 1
  """
  if value > threshold:
    return 0
  else:
    return 1
  
def get_cd_adj(gdf, cd_adj_path, q):
  """
  Description: 
    used to get the centroid distance adjacency matrix of Manhattan zones
  Parameters:
    gdf --- the zones geodataframe
    cd_adj_path --- the centroid distance adjacency matrix path
    q --- the quantile number of distance, which controls the threshold of distance  matrix
  Outputs:
    a csv file --- which store the centroid distance adjacency matrix of Manhattan zones
  """
  target_crs = from_epsg(2263) # convert into New York State Plane Coordinate System
  gdf = gdf.to_crs(target_crs)
  gdf['centroid'] = gdf.centroid # the centorid
  id_list = gdf.index # get the id list
  centroids = np.array(gdf['centroid'].apply(lambda point: (point.x, point.y)).tolist()) # transfer centroids into numpy arrays
  cd_matrix = distance_matrix(centroids, centroids) # get the distance matrix
  cd_matrix_df = pd.DataFrame(cd_matrix,index=id_list,columns=id_list) # get the dataframe
  ds = cd_matrix_df.stack()
  ds = ds[ds!=0]
  dis_threshold = ds.quantile(q) # get the threshold
  cd_adj_df = cd_matrix_df.applymap(lambda x: compare_and_assign(x, dis_threshold)) # build adjacency matrix
  cd_adj_df.to_csv(cd_adj_path) # output
  
def get_d_adj(gdf, d_adj_path, q):
  """
  Description: 
    used to get the spatial distance adjacency matrix of Manhattan zones
  Parameters:
    gdf --- the zones geodataframe
    d_adj_path --- the centroid distance adjacency matrix path
    q --- the quantile number of distance, which controls the threshold of distance  matrix
  Outputs:
    a csv file --- which store the centroid distance adjacency matrix of Manhattan zones
  """
  target_crs = from_epsg(2263) # convert into New York State Plane Coordinate System
  gdf = gdf.to_crs(target_crs)
  id_list = gdf.index # get the id list
  distance_matrix_df = pd.DataFrame(index=gdf['location_id'], columns=gdf['location_id'])
  for i in id_list:
    for j in id_list:
        distance = gdf.loc[i, 'geometry'].distance(gdf.loc[j, 'geometry'])
        distance_matrix_df.at[gdf.loc[i, 'location_id'], gdf.loc[j, 'location_id']] = distance
  ds = distance_matrix_df.stack()
  ds = ds[ds!=0]
  dis_threshold = ds.quantile(q) # get the threshold
  d_adj_df = distance_matrix_df.applymap(lambda x: compare_and_assign(x, dis_threshold)) # build adjacency matrix
  d_adj_df.index.name = 'id'
  d_adj_df.to_csv(d_adj_path) # output  
  
def get_adjacent_rows(row,gdf,sindex):
  """
  Description: 
    used to find the adjacency rows of one row
  Parameters:
    row --- query row in geodatframe
    gdf --- the zones geodataframe
    sindex --- the spatial index
  Returns:
    adjacent_rows --- the adjacency rows
  """
  possible_matches_index = list(sindex.intersection(row.geometry.bounds))
  possible_matches = gdf.iloc[possible_matches_index]
  adjacent_rows = possible_matches[possible_matches.intersects(row.geometry)].location_id.tolist()
  adjacent_rows.remove(row.location_id)
  return adjacent_rows

def get_adj(gdf,adj_path):
  """
  Description: 
    used to get zones adjacency matrix
  Parameters:
    gdf --- the zones geodataframe
    adj_path --- the output path
  Outputs:
    a csv file of zones adjacency matrix
  """
  sindex = gdf.sindex # greate space index
  # Calculate the adjacent relationship
  gdf['adjacent_ids'] = gdf.apply(get_adjacent_rows, args = (gdf,sindex), axis=1)
  # Create the adjacency matrix
  adjacency_matrix = pd.DataFrame(0, index=gdf['location_id'], columns=gdf['location_id'])
  for idx, row in gdf.iterrows():
    adjacency_matrix.loc[row['location_id'], row['adjacent_ids']] = 1 
  for idx, row in adjacency_matrix.iterrows():
    row[idx] = 1
  adjacency_matrix.loc['202','229'] = 1
  adjacency_matrix.loc['202','140'] = 1
  adjacency_matrix.loc['229','202'] = 1
  adjacency_matrix.loc['140','202'] = 1
  adjacency_matrix.loc['194','74'] = 1
  adjacency_matrix.loc['74','194'] = 1
  adjacency_matrix.loc['153','127'] = 1
  adjacency_matrix.loc['153','128'] = 1
  adjacency_matrix.loc['128','153'] = 1
  adjacency_matrix.loc['127','153'] = 1
  adj_matrix = adjacency_matrix.values
  sum = adj_matrix.sum(axis=1)
  s = adj_matrix/sum
  a1 = adj_matrix.dot(s)
  # a2 = a1.dot(s)
  # a_hat = adj_matrix + a1 + a2
  a_hat = adj_matrix + a1
  a_hat[a_hat>0] = 1
  a_hat[a_hat==0] = 0
  final_adj = pd.DataFrame(a_hat,index=gdf['location_id'], columns=gdf['location_id'])
  final_adj.index.name = 'id'
  final_adj.to_csv(adj_path)
  
def read_poi(raw_poi_json_path, poi_json_path):
  """
  Description: 
    used to read original poi file
  Parameters:
    raw_poi_json_path --- original poi geojson path
    poi_json_path --- processed poi geojson path
  Outputs:
    a processed geojson file
  """
  poi_gdf = gpd.read_file(raw_poi_json_path)
  m_poi = poi_gdf['borough'].apply(lambda x:x=='1')
  m_poi_gdf = poi_gdf[m_poi]
  m_poi_gdf.to_file(poi_json_path, driver='GeoJSON')
  
def load_poi(poi_json_path):
  """
  Description: 
    used to read the poi geojson file and output a geo dataframe of poi
  Parameters:
    poi_json_path --- the input poi geojson path
  Returns:
    gdf --- a geodatframe
  """
  gdf = gpd.read_file(poi_json_path)
  return gdf

def find_zone(point,gdf):
  """
  Description: 
    used to find the zone where a point locate
  Parameters:
    point --- the input POINT Object
    gdf --- the zones geodataframe
  Returns:
    zone's id
  """
  for index,zone in gdf.iterrows():
    if point.within(zone['geometry']):
      return zone['location_id']
  return 1000 # not in any zone

def get_poi_vec(id,poi_gdf):
  """
  Description: 
    used to get the poi vector of a zone
  Parameters:
    id --- the input zone id
    poi_gdf --- the poi geodataframe
  Returns:
    poi_list --- the poi vector
  """
  poi_list = np.zeros(13)
  loc_gdf = poi_gdf[poi_gdf['zone_id'].apply(lambda x:x==id)]
  value_counts = loc_gdf['facility_t'].value_counts()
  for i, v in value_counts.items():
    poi_list[int(i)-1] = v
  return poi_list

def get_poi_feature(poi_json_path,zone_json_path,poi_feature_path):
  """
  Description: 
    used to get the poi feature of each zone
  Parameters:
    poi_json_path --- the poi geojson path
    zone_json_path --- the zones geojson path
    poi_feature_path --- the output poi feature csv path
  Outputs:
    a csv file
  """
  # load to geopandas dataframe
  poi_gdf = load_poi(poi_json_path)  
  zone_gdf = load_zones(zone_json_path)
  poi_gdf['zone_id'] = poi_gdf['geometry'].apply(find_zone, gdf=zone_gdf) # apply poi points into zones
  zone_gdf['poi_vec'] = zone_gdf['location_id'].apply(get_poi_vec, poi_gdf=poi_gdf) # final poi vector
  poi_df = zone_gdf['poi_vec'].apply(pd.Series) # expand features
  # Rename the new columns
  column_names = [f'poi_{i+1}' for i in range(poi_df.shape[1])]
  poi_df.columns = column_names
  poi_df = poi_df.apply(lambda x: (x-np.min(x))/(np.max(x)-np.min(x))) # normalise
  poi_df.to_csv(poi_feature_path)
  
def adj_fussion(adj_path,d_adj_path,cd_adj_path,w_adj_path):
  """
  Description: 
    used to fuse three adjacency matrix
  Parameters:
    adj_path --- the zones adjacency matrix path
    d_adj_path --- the zones space adjacency matrix path
    cd_adj_path --- the zones centroid adjacency matrix path
    w_adj_path --- the output weighted adjacency matrix path
  Outputs:
    a fused adj matrix csv file
  """
  adj = pd.read_csv(adj_path,index_col=0)
  d_adj = pd.read_csv(d_adj_path,index_col=0)
  cd_adj = pd.read_csv(cd_adj_path,index_col=0)
  w_adj = adj + d_adj + cd_adj
  w_adj[w_adj>0] = 1
  w_adj[w_adj==0] = 0
  w_adj.to_csv(w_adj_path)
