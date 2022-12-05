from typing import Tuple
import numpy as np
from numpy.typing import NDArray


def _aabb_volume(aabb: NDArray[np.float32]) -> float:
    return float(np.prod(aabb[1] - aabb[0]))

def aabb_intersection_ratios(aabb1: NDArray[np.float32], aabb2: NDArray[np.float32]) -> Tuple[float, float, float]:
    """For two axis-aligned 3D bounding boxes, compute the ratio of the volume of intersection over the volume of union and the volume of the two bounding boxes.
    Args:
        aabb1: A 2D array of shape (2, 3) representing the first axis-aligned
            bounding box.
        aabb2: A 2D array of shape (2, 3) representing the second axis-aligned
            bounding box.

    Returns:
        (IoU, intersection / aabb1 volume, interesection / aabb2 voluje)
    """
    if (aabb1[0] > aabb1[1]).any() or (aabb2[0] > aabb2[1]).any():
        raise ValueError("The first element of the bounding box should be smaller than the second element.")

    # Compute the intersection of the two bounding boxes.
    intersection_volume = aabb_intersection_volume(aabb1, aabb2)

    # Compute the volume of the two bounding boxes.
    aabb1_volume = _aabb_volume(aabb1)
    aabb2_volume = _aabb_volume(aabb2)

    # Compute the union of the two bounding boxes.
    union_volume = aabb1_volume + aabb2_volume - intersection_volume

    # Compute the intersection over union.
    iou = float(intersection_volume / union_volume)

    assert 0 <= iou <= 1, f"for some reson IoU is {iou}. It should be between 0 and 1."

    return iou, intersection_volume / aabb1_volume, intersection_volume / aabb2_volume

def aabb_iou(aabb1: NDArray[np.float32], aabb2: NDArray[np.float32]) -> float:
    """Compute the intersection over union of two axis-aligned 3D bounding boxes.

    Args:
        aabb1: A 2D array of shape (2, 3) representing the first axis-aligned
            bounding box.
        aabb2: A 2D array of shape (2, 3) representing the second axis-aligned
            bounding box.

    Returns:
        The intersection over union of the two axis-aligned bounding boxes.
    """
    return aabb_intersection_ratios(aabb1, aabb2)[0]

def aabb_intersection_volume(aabb1: NDArray[np.float32], aabb2: NDArray[np.float32]) -> float:
    """Compute the intersection of two axis-aligned 3D bounding boxes.

    Args:
        aabb1: A 2D array of shape (2, 3) representing the first axis-aligned
            bounding box.
        aabb2: A 2D array of shape (2, 3) representing the second axis-aligned
            bounding box.

    Returns:
        The intersection of the two axis-aligned bounding boxes.
    """
    # Compute the intersection of the two bounding boxes.
    if (aabb1[0] > aabb1[1]).any() or (aabb2[0] > aabb2[1]).any():
        raise ValueError("The first element of the bounding box should be smaller than the second element.")

    intersection = np.maximum(0, np.minimum(aabb1[1], aabb2[1]) - np.maximum(aabb1[0], aabb2[0]))

    volume = np.prod(intersection)

    assert volume >= 0, "intersection volume has to be positive"
    return volume