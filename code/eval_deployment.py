from pathlib import Path 
import geopandas as gpd 
import matplotlib as mpl
from matplotlib import pyplot as plt

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
    
    return captured_gdf.shape[0], missed_gdf.shape[0]

def plot(data, title, yticks, filename):
    plt.rcParams['font.size'] = 16
    
    def plot_function(x, ax):
        ax = graph[x]
        ax.set_xlabel(x, fontstyle='italic')
        return data.xs(x).plot(kind='bar', stacked='True', ax=ax, legend=False, width=0.8)

    n_subplots = len(data.index.levels[0])
    fig, axes = plt.subplots(nrows=1, ncols=n_subplots, sharey=False, figsize=(9, 10))  # width, height

    graph = dict(zip(data.index.levels[0], axes))
    plots = list(map(lambda x: plot_function(x, graph[x]), graph))
    
    axes[0].set_yticks(yticks)
    axes[1].set_yticks(yticks)

    axes[0].spines[['right']].set_visible(False)
    axes[1].spines[['left']].set_visible(False)
    axes[1].set_yticks([])

    for idx, c in enumerate(axes[0].containers):
        if idx == 0:
            # N/A label for Enverus
            bar1_data = c.datavalues
            axes[0].bar_label(c, padding=0, labels=[f'N/A' for x in c.datavalues])
        elif idx == 1:
            # N/A label for Enverus
            axes[0].bar_label(c, padding=0, labels=[f'N/A' for x, x0 in zip(c.datavalues, bar1_data)])
    for idx, c in enumerate(axes[1].containers):
        if idx == 0:
            bar1_data = c.datavalues
            axes[1].bar_label(c, padding=0, labels=[f'{x:,.0f}' for x in c.datavalues])
        elif idx == 1:
            axes[1].bar_label(c, padding=0, labels=[f'{(x+x0):,.0f}' for x, x0 in zip(c.datavalues, bar1_data)])

    axes[0].set_xticks(axes[0].get_xticks())
    axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=0)
    axes[1].set_xticks(axes[1].get_xticks())
    axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=0)

    axes[0].yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))

    fig.subplots_adjust(wspace=0)
    axes[0].legend(loc='upper left')
    fig.supylabel("Count", x=0.0)
    fig.supxlabel("Dataset + Well pad type", y=0.01)
    fig.suptitle("", y=0.91)
    plt.title(title, x=0)
    plt.show()
    plt.savefig(filename)


def eval_deployment():
    print("Evaluating well pad deployment detections...")
    data = {}
    for basin in ['permian', 'denver']:
        data[basin] = {
            'Captured': {
                ('Enverus', 'Active'): 0,
                ('Enverus', 'All'): 0,
            },
            'Missed': {
                ('Enverus', 'Active'): 0,
                ('Enverus', 'All'): 0,
            }
        }
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

        for active_only, descr in [(True, "Active"), 
                                   (False, "All")]:
            num_captured, num_missed = evaluate(deploy_gdf, reported_gdf, active_only)
            data[basin]['Captured'][('HIFLD', descr)] = num_captured
            data[basin]['Missed'][('HIFLD', descr)] = num_missed

    plot(pd.DataFrame(data['permian']), 
         "Permian Basin", 
         range(0, 200000, 25000),
         'results/fig2_permian.png')
    plot(pd.DataFrame(data['denver']), 
         "Denver Basin", 
         range(0, 30000, 5000),
         'results/fig2_denver.png')
    print("Results saved to [results] directory.")



