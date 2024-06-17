# well-pad-denver-permian
Public repository for "Deep learning for detecting and characterizing oil and gas well pads in satellite imagery."

This repository contains: 
1. Datasets used to train well pad and storage tank detection models (`data/training/datasets`). We note that we are unable to redistribute the satellite imagery used to train the models in this study due to data licensing. 
2. Well pad and storage tank deployment detections across the entire Permian and Denver basins generated in this study (`data/deployment`).
3. Outputs from experiments on training dataset test splits (`data/training/results`), and evaluation code   (`code/eval_test.py`) for replicating performance metrics (i.e., Tables 1 and 4 in the paper).
4. Evaluation code (`code/eval_deployment.py`) for comparing deployment detections to reported HIFLD well pad data (i.e. Fig. 2 in the paper). 

Results from 3. and 4. can be verified by running the evaluation code using the command `python code/eval_all.py`.

Code for training the models in the study may be made available upon request to the corresponding author. 
