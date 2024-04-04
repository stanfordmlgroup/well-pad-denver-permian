import pandas as pd
from detectron2.structures import Instances, Boxes
import torch
from collections import defaultdict
import numpy as np

from eval import DetectionEvaluator
from util.constants import *


def evaluate_meta(meta, basin, config):
             
    for col in ['pred_boxes', 'scores', 'gt_boxes']:
        meta[col] = meta[col].map(eval)
    
    preds, gts = [], []
    
    for image_id, pred_boxes, scores, gt_boxes in zip(meta['image_id'], 
                                                      meta['pred_boxes'], 
                                                      meta['scores'],
                                                      meta['gt_boxes']):
        
        pred_instances = Instances(
            image_size=(IMG_SIZE, IMG_SIZE),
            pred_boxes=Boxes(pred_boxes),
            scores=torch.tensor(scores),
            pred_classes=torch.tensor([0 for _ in pred_boxes]),
        )
        preds.append({
            'image_id': image_id, 
            'instances': pred_instances
        })

        gt_instances = Instances(
            image_size=(IMG_SIZE, IMG_SIZE), 
            gt_boxes=Boxes(gt_boxes),
            gt_classes=torch.tensor([0 for _ in gt_boxes]),
        )
        gts.append({
            'image_id': image_id, 
            'instances': gt_instances, 
        }) 

    detection_evaluator = DetectionEvaluator(
        iou_thresh=config['iou_threshold'],
        conf_threshold=config['conf_threshold'][basin],
    )
    detection_evaluator.process(gts, preds)
    metrics = detection_evaluator.evaluate()
    return metrics

def evaluate_storage_tanks(basin):
    meta = pd.read_csv(RESULTS_DIR / 'storage-tank' / f'{basin}_test_set.csv')
    basin_metrics = evaluate_meta(meta, basin, EVAL_CONFIG['storage_tank'])
    return basin_metrics

def evaluate_well_pads(basin):
    basin_metrics = defaultdict(list)
    meta_paths = (RESULTS_DIR / "well-pad" / basin).glob("*.csv")
    for meta_path in meta_paths:
        meta = pd.read_csv(meta_path)
        for k, v in evaluate_meta(meta, basin, EVAL_CONFIG['well_pad']).items():
            basin_metrics[k].append(v)
    return basin_metrics

def evaluate_all():    
    print("*"*80) 
    print("Well Pad Detection Pipeline -- Test set results (Table 1)")
    permian_wp = evaluate_well_pads('permian')
    denver_wp = evaluate_well_pads('denver')
    print('Permian:')
    for metric in METRICS:
        metric_vals = permian_wp[metric]
        print(f"{metric}: {np.mean(metric_vals)} ± {np.std(metric_vals)}")
    print()
    print('Denver:')
    for metric in METRICS:
        metric_vals = denver_wp[metric]
        print(f"{metric}: {np.mean(metric_vals)} ± {np.std(metric_vals)}")
    print()
    print("Overall:")
    for metric in METRICS:
        permian_metric_vals = permian_wp[metric]
        denver_metric_vals = denver_wp[metric]
        overall_metric_mean = (
            np.mean(permian_metric_vals) * (N_TEST_WP['permian'] / (N_TEST_WP['permian'] + N_TEST_WP['denver'])) + 
            np.mean(denver_metric_vals) * (N_TEST_WP['denver'] / (N_TEST_WP['permian'] + N_TEST_WP['denver']))
        )
        overall_metric_std = (
            np.std(permian_metric_vals) * (N_TEST_WP['permian'] / (N_TEST_WP['permian'] + N_TEST_WP['denver'])) + 
            np.std(denver_metric_vals) * (N_TEST_WP['denver'] / (N_TEST_WP['permian'] + N_TEST_WP['denver']))
        )
        print(f"{metric}: {overall_metric_mean} ± {overall_metric_std}")
    print("""
    Note: Metrics displayed here are generated from 10 stochastic 
    runs separate from those used in the paper and closely (but not exactly)
    match the paper results.\n
    """)
    print("*"*80) 
    print("Storage Tank Detection -- Test set results (Table 4)")
    permian_st = evaluate_storage_tanks('permian')
    denver_st = evaluate_storage_tanks('denver')
    print("Permian:")
    for metric in METRICS:
        print(f"{metric}: {permian_st[metric]}")
    print()
    print("Denver")
    for metric in METRICS:
        print(f"{metric}: {denver_st[metric]}")
    print()
    print("Overall:")
    for metric in METRICS:
        permian_metric_val = permian_st[metric]
        denver_metric_val = denver_st[metric]
        overall_metric_val = (
            permian_metric_val * (N_TEST_ST['permian'] / (N_TEST_ST['permian'] + N_TEST_ST['denver'])) +
            denver_metric_val * (N_TEST_ST['denver'] / (N_TEST_ST['permian'] + N_TEST_ST['denver']))
        )
        print(f"{metric}: {overall_metric_val}")
    print("*"*80) 
    
evaluate_all()