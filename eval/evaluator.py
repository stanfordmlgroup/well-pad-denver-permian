import numpy as np

from . import detection
# from util import constants as C


class DatasetEvaluator:
    """
    Base class for a dataset evaluator.

    This class will accumulate information of the inputs/outputs (by :meth:`process`),
    and produce evaluation results in the end (by :meth:`evaluate`).
    """

    def reset(self):
        """
        Preparation for a new round of evaluation.
        Should be called before starting a round of evaluation.
        """
        raise NotImplementedError(
            "[reset] method need to be implemented in child class.")

    def process(self, ground_truths, predictions):
        """
        Process the pair of ground_truths and predictions.
        If they contain batches, the pairs can be consumed one-by-one using `zip`:

        Args:
            ground_truths (list): the ground_truth values (annotations, labels, etc.)
            predictions (list): the model predictions (probabilities, predictions, etc.)
        """
        raise NotImplementedError(
            "[process] method need to be implemented in child class.")

    def evaluate(self):
        """
        Evaluate/summarize the performance, after processing all ground_truth/prediction pairs.

        """
        raise NotImplementedError(
            "[evaluate] method need to be implemented in child class.")


class DetectionEvaluator(DatasetEvaluator):
    """
    Evaluator for detection task.
    This class will accumulate information of the ground_truth/prediction (by :meth:`process`),
    and produce evaluation results in the end (by :meth:`evaluate`).
    """

    def __init__(
            self,
            iou_thresh=0.5,
            conf_threshold=None):
       
        self._evaluator = detection.Evaluator()
        self._iou_thresh = iou_thresh
        self._conf_threshold = conf_threshold
        self._round = round
        self.reset()

    def reset(self):
        self._bbox = detection.BoundingBoxes()

    def process(self, groudtruths, predictions):
        """
        Inputs format:
        https://detectron2.readthedocs.io/en/latest/tutorials/models.html?highlight=input%20format#model-input-format
        Outputs format:
        https://detectron2.readthedocs.io/en/latest/tutorials/models.html?highlight=input%20format#model-output-format
        """
        for sample_input, sample_output in zip(groudtruths, predictions):
            image_id = sample_input['image_id']
            gt_instances = sample_input['instances']
            pred_instances = sample_output['instances']
            width = -1
            height = -1
            for i in range(len(gt_instances)):
                instance = gt_instances[i]
                class_id = instance.get(
                    'gt_classes').cpu().detach().numpy().item()
                boxes = instance.get('gt_boxes')
                height, width = instance.image_size
                for box in boxes:
                    box_np = box.cpu().detach().numpy()
                    
                    bb = detection.BoundingBox(
                        image_id,
                        class_id,
                        box_np[0],
                        box_np[1],
                        box_np[2],
                        box_np[3],
                        detection.CoordinatesType.Absolute,
                        (width,
                         height),
                        detection.BBType.GroundTruth,
                        format=detection.BBFormat.XYX2Y2)
                    self._bbox.addBoundingBox(bb)
            for i in range(len(pred_instances)):
                instance = pred_instances[i]
                height, width = instance.image_size
                class_id = instance.get(
                    'pred_classes').cpu().detach().numpy().item()
                scores = instance.get('scores').cpu().detach().numpy().item()
                boxes = instance.get('pred_boxes')
                for box in boxes:
                    box_np = box.cpu().detach().numpy()
                    bb = detection.BoundingBox(
                        image_id,
                        class_id,
                        box_np[0],
                        box_np[1],
                        box_np[2],
                        box_np[3],
                        detection.CoordinatesType.Absolute,
                        (width,
                         height),
                        detection.BBType.Detected,
                        scores,
                        format=detection.BBFormat.XYX2Y2)
                    self._bbox.addBoundingBox(bb)

    def evaluate(self):

        results = self._evaluator.GetPascalVOCMetrics(
            self._bbox, self._iou_thresh)
        if isinstance(results, dict):
            results = [results]
        
        result = results[0]
        metrics = {}
        precision = np.array(result['precision'])
        recall = np.array(result['recall'])
        score = np.array(result['score'])            
            
        # preset threshold
        if self._conf_threshold is not None:
            for ind_s, s in enumerate(score):
                if s < self._conf_threshold:
                    metrics[f'Precision'] = precision[ind_s]
                    metrics[f'Recall'] = recall[ind_s]
                    break
            else:
                metrics[f'Precision'] = -1.
                metrics[f'Recall'] = -1.

        metrics['AP'] = result['AP']
        return metrics