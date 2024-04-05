## Data

### Training datasets and experiment results
- `training/datasets`: Datasets used to train detection models. Note that `annotations_image` for each example are with respect to a image downloaded from the Google Earth satellite basemap at zoom level
1600 and EPSG:3857, with sizes 640x640px and 512x512px for well pads and storage tanks respectively 
(see also the `image_extent` column). `annotations_latlon` also provides the annotations in coordinate space. 
- `training/results`: Model predictions on the test splits of the well pad and storage tank datasets used to replicate paper results (`code/eval_test.py`). For well pads, predictions from 10 runs because stochasticity in the results to due to random data augmentations (storage tanks results are deterministic.)

### Deployment 
- `deployment`: Deployment detections for well pads and storage tanks across the entire Permian and Denver basins. Datasets contain confidence scores (`bbox_score`) and coordinate locations (`geometry`) for each detection, as well as a well pad identifier (`wp_id`) that indicates which well pad a storage tank belongs to. 
- `deployment/well-pad/reported`: Contains HIFLD well data in the Permian and Denver basins clustered into well pads (see Supplementary Note 1). Used to assess the deployment detections (`code/eval_deployment.py`). Note that the Enverus data and any metrics from our assessment against it are not provided because the data source is private.



