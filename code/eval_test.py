import pandas as pd
from detectron2.structures import Instances, Boxes
import torch
from collections import defaultdict
import numpy as np
from tabulate import tabulate

from eval import DetectionEvaluator
from util import *


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

def evaluate_storage_tank_basin(basin):
    meta = pd.read_csv(RESULTS_DIR / 'storage-tank' / f'{basin}_test_set.csv')
    basin_metrics = evaluate_meta(meta, basin, EVAL_CONFIG['storage_tank'])
    return basin_metrics

def evaluate_well_pad_basin(basin):
    basin_metrics = defaultdict(list)
    meta_paths = (RESULTS_DIR / "well-pad" / basin).glob("*.csv")
    for meta_path in meta_paths:
        meta = pd.read_csv(meta_path)
        for k, v in evaluate_meta(meta, basin, EVAL_CONFIG['well_pad']).items():
            basin_metrics[k].append(v)
    return basin_metrics

def evaluate_well_pads():
    n_permian, n_denver = N_TEST_WP['permian'], N_TEST_WP['denver']
    n_total = n_permian + n_denver
    means, stds = [], []
    for basin in ['permian', 'denver']:
        trial_metrics = evaluate_well_pad_basin(basin)
        means.append([np.mean(trial_metrics[metric]) for metric in METRICS])
        stds.append([np.std(trial_metrics[metric]) for metric in METRICS])
    
    means = np.array(means)
    stds = np.array(stds)
    overall_means = np.average(
        means, axis=0, weights=[n_permian/n_total, n_denver/n_total]
    )
    overall_stds = np.average(
        stds, axis=0, weights=[n_permian/n_total, n_denver/n_total]
    )
    mean_df = pd.DataFrame(
        np.vstack([means, overall_means])
    ).round(3).astype(str)
    std_df = pd.DataFrame(
        np.vstack([stds, overall_stds])
    ).round(3).astype(str)
    df = pd.DataFrame(mean_df + " ± " + std_df)
    df.columns = METRICS
    df.index = ['Permian', 'Denver', 'Overall']
    with open('results/table1.txt', 'w') as f:
        f.write(
            tabulate(df, headers='keys', tablefmt='fancy_grid')
        )

def evaluate_storage_tanks():
    n_permian, n_denver = N_TEST_ST['permian'], N_TEST_ST['denver']
    n_total = n_permian + n_denver
    metrics = []
    for basin in ['permian', 'denver']:
        basin_metrics = evaluate_storage_tank_basin(basin)
        metrics.append([basin_metrics[metric] for metric in METRICS])
        
    metrics = np.array(metrics)    
    overall_metrics = np.average(
        metrics, axis=0, weights=[n_permian/n_total, n_denver/n_total]
    )
    df = pd.DataFrame(
        np.vstack([metrics, overall_metrics]),
        columns=METRICS,
        index=['Permian', 'Denver', 'Overall']
    ).round(3)
    with open('results/table4.txt', 'w') as f:
        f.write(
            tabulate(df, headers='keys', tablefmt='fancy_grid')
        )
    
def eval_test():
    print("Evaluating well pad test set...")
    evaluate_well_pads()
    print("Evaluating storage tank test set...")
    evaluate_storage_tanks()
    print("Results saved to [results] directory.")