import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
from shapely.geometry import Point, LineString

def region_vis(tripdata):
    gdf = gpd.read_file('./data/NYC Taxi Zones.geojson') # the taxi zones
    centroid = gdf.centroid # get the centroid
    zones = gdf['zone'] 
    # get each OD pair's coordinate
    tripdata['PUPoint'] = tripdata['PULocationID'].apply(lambda x:centroid[x-1])
    tripdata['DOPoint'] = tripdata['DOLocationID'].apply(lambda x:centroid[x-1])
    # start draw
    ax = gdf.plot(figsize=(100, 100), alpha=0.5, edgecolor='k') # plot the region
    
    # Create a new GeoDataFrame containing the centroids and names
    centroid_gdf = gpd.GeoDataFrame({'name': zones, 'geometry': centroid}, crs=gdf.crs)
    # Plot the centroid_gdf as a dot on the same plot, with names as labels
    centroid_gdf.plot(ax=ax, color='red', markersize=50)
    for idx, row in centroid_gdf.iterrows():
        ax.annotate(row['name'], xy=row['geometry'].coords[0], ha='center', va='center', color='black')
           
    # Create a list of LineString geometries from the start and end points
    lines = [LineString([Point(start), Point(end)]) for start, end in tripdata[['PUPoint', 'DOPoint']].values]
    # Create a GeoDataFrame containing the lines
    lines_gdf = gpd.GeoDataFrame(geometry=lines, crs=gdf.crs)
    lines_gdf.plot(ax=ax)
    
    # Show the plot
    plt.show()
    
def bar_plot(data, period, size):
    # Create the bar plot
    plt.figure(figsize=size)
    plt.plot(data.index, data['count'])

    # Set the plot title and axis labels
    plt.title('Count by ' + period)
    plt.xticks(rotation=90)
    plt.xlabel(period)
    plt.ylabel('Count')

    # Show the plot
    plt.show()
    
def data_heatmap(data):
    # create a new dataframe with all possible combinations of PULocationID and DOLocationID
    heatmap_df = data.pivot(index='PULocationID', columns='DOLocationID', values='count')
    heatmap_df = heatmap_df.reindex(range(1, 264), range(1, 264), fill_value=0)

    # create heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(heatmap_df, cmap='YlGnBu')
    plt.title('Heatmap of Taxi Pick-up and Drop-off Locations')
    plt.xlabel('Drop-off Location')
    plt.ylabel('Pick-up Location')
    plt.show()