from arkitekt_next import easy, register
from mikro_next.api.schema import ROI, from_array_like, Image
from kraph.api.schema import (
    create_structure_metric,
    MetricKind,
    Metric,
    Graph,
)
import numpy as np


@register
def measure_maximum(image: Image, graph: Graph) -> Metric:
    metric = create_structure_metric(
        structure=image,
        label="max",
        value=float(np.max(image.data)),
        graph=graph,
        metric_kind=MetricKind.FLOAT,
    )

    return metric


@register
def measurem_min(image: Image, graph: Graph) -> Metric:
    metric = create_structure_metric(
        structure=image,
        label="min",
        value=float(np.min(image.data)),
        graph=graph,
        metric_kind=MetricKind.FLOAT,
    )

    return metric


@register
def measure_mean(image: Image, graph: Graph) -> Metric:
    metric = create_structure_metric(
        structure=image,
        label="mean",
        value=float(np.mean(image.data)),
        graph=graph,
        metric_kind=MetricKind.FLOAT,
    )

    return metric


@register
def measure_roi_area(roi: ROI, graph: Graph) -> Metric:
    """Measure the area of a Region of Interest (ROI)."""
    # Calculate area based on ROI vectors
    area = 0

    if roi.vectors and len(roi.vectors) > 0:
        # For polygon ROIs, calculate area using shoelace formula
        if len(roi.vectors) >= 3:
            vectors = roi.vectors
            n = len(vectors)
            area = 0.0

            for i in range(n):
                j = (i + 1) % n
                area += vectors[i][4] * vectors[j][3]
                area -= vectors[j][4] * vectors[i][3]

            area = abs(area) / 2.0
        else:
            # For simple shapes like rectangles (2 points defining corners)
            if len(roi.vectors) == 2:
                v1, v2 = roi.vectors[0], roi.vectors[1]
                width = abs(v2.x - v1.x)
                height = abs(v2.y - v1.y)
                area = width * height

    metric = create_structure_metric(
        structure=roi,
        label="area",
        value=float(area),
        graph=graph,
        metric_kind=MetricKind.FLOAT,
    )

    return metric
