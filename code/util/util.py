import pandas as pd 
import geopandas as gpd
from shapely import wkt

def csv2gdf(path):
    df = pd.read_csv(path)
    df['geometry'] = df['geometry'].map(wkt.loads)
    return gpd.GeoDataFrame(df, geometry='geometry', crs=4326)