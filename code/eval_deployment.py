from pathlib import Path 
import geopandas as gpd 

from util import *

def evaluate(deploy_gdf,
             reported_gdf,
             active_only=False,
             buffer_m=50,
             crs=LOCAL_EPSG):
    
    if active_only:
        reported_gdf = reported_gdf.query("Active")        
    
    deploy_gdf = deploy_gdf.to_crs(crs)
    deploy_gdf.geometry = deploy_gdf.geometry.buffer(buffer_m)
    deploy_gdf = deploy_gdf.to_crs(4326)

    captured_gdf = gpd.sjoin(reported_gdf, deploy_gdf, predicate='within')
    # Some reported_gdf well pads (points) appear in multiple deploy_gdf boxes. Deduplicate.
    captured_gdf = captured_gdf.drop_duplicates(['Latitude', 'Longitude'])
    missed_gdf = reported_gdf.drop(captured_gdf.index)
    
    return {
        '# Reported Well Pads': reported_gdf.shape[0],
        '# Captured Well Pads': captured_gdf.shape[0],
        '# Missed Well Pads': missed_gdf.shape[0] 
    }

def evaluate_all():
    
    for basin in ['permian', 'denver']:
        deploy_gdf = csv2gdf(
            DEPLOYMENT_DIR / 
            'well-pad' / 
            f'{basin}_well_pads.csv'
        )         
        reported_gdf = csv2gdf(
            DEPLOYMENT_DIR / 
            'well-pad' / 
            'reported' / 
            f'{basin}_hifld.csv'
        )
        print(basin.capitalize())
        print("*"*50)
        print(f"# Well Pad Detections: {deploy_gdf.shape[0]}")
        print("*"*50)
        for active_only, descr in [(True, "Active Reported Well Pads (HIFLD)"), 
                                   (False, "All Reported Well Pads (HIFLD)")]:
            print(descr)
            metrics = evaluate(deploy_gdf, reported_gdf, active_only)
            for k, v in metrics.items():
                print(f"{k}: {v}")
            print("*"*50)
        print()
        
evaluate_all()



