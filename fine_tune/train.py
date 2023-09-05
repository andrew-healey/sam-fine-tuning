"""
Setup:
- will be passed a Roboflow dataset ID - for now, assume it's an rf download.
- should train a mask decoder which can serve as both normal SAM and custom SAM
- no cost-saving or time-saving measures for now - ie batching, early stopping
- strictly scoped to projects with >40 images and >200 annotations
- should include SAM-fallback logic -- i.e. stability score/iou prediction/unsure of cls -> kick back to SAM
- macro threshold for using model:
    - if avg # of clicks per instance is < than SAM (aka model is mask-good), OR
    - if all classes have >75% on-axis of the confusion matrix (aka model is cls-good)
- micro threshold for using a mask&cls prediction:
    - model is mask-good AND cls-good
    - highest cls iou prediction is higher than for SAM prediction(s?), AND
    - user has enabled "use cls predictions" in the UI
- micro threshold for using a mask-only prediction:
    - model is mask-good
    - highest cls iou prediction is higher than for SAM prediction(s?)
- micro threshold for using a cls prediction:
    - model is cls-good
    - user has enabled "use cls predictions" in the UI
"""

"""
Todos:
- Clicks-per-instance benchmarking
- Confusion matrix benchmarking
- Automatically pick best pred-IoU threshold for SAM vs. custom SAM
- Attn masks + duplicate points: LHS can't attend to RHS, vice versa
- Simple Flask server for requesting+downloading a trained model
- ONNX export as a function
- Train loop as a function - no custom configs yet, just set defaults
- Model-ify the configurable encoder & decoder
"""

#
# Metrics
#

def confusion_matrix(predictor: nn.Module, valid_dataset: Dataset) -> torch.Tensor:
    """
    Returns a confusion matrix of shape (num_classes, num_classes).
    The rows are the ground truth classes, the columns are the predicted classes.
    """

    running_matrix = torch.zeros((predictor.mask_decoder.num_classes, predictor.num_classes), dtype=torch.int32)

    for 