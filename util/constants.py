from pathlib import Path 

IMAGE_SIZE=512
RESULTS_DIR = Path("data/training/results/")
DEPLOYMENT_DIR = Path("data/deployment/")

EVAL_CONFIG = {
    'storage_tank': {
        'iou_threshold': 0.5,
        'conf_threshold': {
            'permian': 0.9093543291, 
            'denver': 0.9400245547,
        },
    },
    'well_pad': {
        'iou_threshold': 0.3,
        'conf_threshold': {
            'permian': 0.5941267788, 
            'denver': 0.4843277037,
        },
    },
}

N_TEST_WP = {'permian': 3003, 'denver': 2893}
N_TEST_ST = {'permian': 305, 'denver': 237}
METRICS = ['AP', 'Precision', 'Recall']

LOCAL_EPSG = 32613