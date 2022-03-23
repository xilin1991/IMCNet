import numpy as np


def db_eval_mae_multi(annotations, segmentations):
    mae = 0.0
    batch_size = annotations.shape[0]

    for idx in range(batch_size):
        annotation = annotations[idx, 0, :, :]
        segmentation = segmentations[idx, 0, :, :]

        mae += db_eval_mae(annotation, segmentation)

    mae /= batch_size
    return mae


def db_eval_mae(annotation, segmentation):
    annotation = np.asarray(annotation, np.float32)
    segmentation = np.asarray(segmentation, np.float32)

    residual = np.abs(annotation - segmentation)
    mae = np.mean(residual)
    return mae
