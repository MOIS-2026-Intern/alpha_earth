# Seoul Flood Risk Model Validation Review

## Summary

Current selected model: `Hybrid-plus-drainage-gu-stats`

Classifier: Earth Engine `smileGradientTreeBoost`

Purpose: static flood susceptibility / priority-risk mapping, not event-time flood forecasting.

Overall judgment: the current validation design is appropriate for an initial Seoul static risk map. The results are not just random signal: spatial 5-fold performance, top-k concentration, and official urban flood map overlap all point in the same direction. The remaining risks are mainly label uncertainty, Seoul-only generalization, and buffer/fold sensitivity that should be expanded before making broader claims.

## Current Evidence

5-fold spatial CV:

| Metric | Mean | Std |
|---|---:|---:|
| Accuracy | 0.839 | 0.053 |
| Kappa | 0.681 | 0.108 |
| ROC-AUC | 0.911 | 0.045 |
| PR-AUC | 0.901 | 0.045 |
| Recall | 0.835 | 0.074 |
| F1 | 0.838 | 0.057 |

Top-k validation against held-out flood reference points:

| Risk Area | Point Recall | Lift |
|---|---:|---:|
| Top 20% | 0.833 | 4.13 |
| Top 10% | 0.626 | 6.24 |
| Top 5% | 0.406 | 8.27 |

Official urban flood map validation:

| Official Map | Top 20% Recall | Top 20% Lift | Top 5% Recall | Top 5% Lift |
|---|---:|---:|---:|---:|
| 30-year | 0.592 | 2.97 | 0.242 | 4.86 |
| 50-year | 0.593 | 2.97 | 0.244 | 4.89 |
| 80-year | 0.720 | 3.61 | 0.305 | 6.27 |
| 100-year | 0.600 | 3.00 | 0.247 | 4.95 |
| 500-year | 0.617 | 3.10 | 0.247 | 5.09 |
| Historical max | 0.605 | 3.03 | 0.264 | 5.30 |

## Validation Logic Check

### Spatial 5-fold

The fold split is based on longitude/latitude grid blocks (`SPATIAL_BLOCK_DEGREES=0.015`) rather than random pixel splits. This is a good choice because nearby pixels are highly correlated in spatial data. Random splits would likely overestimate performance.

Fold positive counts are uneven: 31, 66, 50, 60, 50. This is acceptable for now, but fold 0 has fewer raw reference points than the others. The model still shows stable enough performance across folds, but this unevenness should be mentioned as a limitation.

Important: AlphaEarth score is recomputed from training positive points for each fold. The validation fold positive points are not used when building the fold-specific AlphaEarth reference mean. This reduces leakage risk.

### Positive / Negative Buffer

Current setting:

- Positive buffer: 60 m
- Negative exclusion buffer around flood points: 300 m
- Positive/negative samples: 200 each

The logic is reasonable. Positive buffer turns point flood records into a small affected area, while negative buffer avoids sampling negative pixels too close to known flood locations.

Fold 0 buffer sensitivity for the selected model:

| Positive Buffer | Negative Buffer | Accuracy | Kappa | ROC-AUC | PR-AUC | F1 |
|---:|---:|---:|---:|---:|---:|---:|
| 30 m | 300 m | 0.822 | 0.619 | 0.912 | 0.782 | 0.740 |
| 60 m | 300 m | 0.850 | 0.700 | 0.923 | 0.916 | 0.856 |
| 90 m | 300 m | 0.840 | 0.680 | 0.924 | 0.921 | 0.845 |
| 60 m | 200 m | 0.855 | 0.710 | 0.915 | 0.888 | 0.862 |
| 60 m | 500 m | 0.858 | 0.710 | 0.918 | 0.904 | 0.860 |

Interpretation: the current 60 m / 300 m setting is defensible. Positive buffer 30 m makes the label stricter and lowers F1/PR-AUC. Positive buffer 90 m remains close to the baseline. Negative buffer 200-500 m does not break performance on fold 0, which suggests the model is not overly dependent on exactly 300 m.

Limitation: this sensitivity check has only been run on fold 0. Before final reporting, repeat it across all 5 folds if time allows.

### Negative Labels

Negative pixels are not guaranteed true non-flood areas. They are sampled from areas away from known flood points and existing water, so they are better described as assumed negatives or unobserved negatives.

This is acceptable for a first susceptibility model, but the report should avoid saying the model learned absolute flooded vs never-flooded truth. A safer expression is: it learned patterns distinguishing known flood-reference areas from sampled background areas.

### Top-k Metrics

Top-k metrics fit the project goal well. Since this model is a static risk map, the most useful question is not only "is each pixel classified correctly?" but "if we inspect the highest-risk 5/10/20% of Seoul, how many known flood points or official flood areas are captured?"

The top-k results are strong:

- Top 20% captures about 83.3% of held-out flood points.
- Top 5% captures about 40.6% of held-out flood points.
- Top 5% is about 8.27 times more efficient than area-random selection.

This supports using the model for priority screening. It does not prove calibrated flood probability.

### Official Flood Map Validation

The official map validation is useful because it checks the model against polygon-based external references, not only the point labels used for training.

The result is consistent: official flood polygons have higher mean predicted probability than outside areas, and top-risk areas overlap official polygons more than random area selection.

Limitations:

- Official frequency maps are scenario/reference maps, not event-date observations.
- Frequency datasets are not perfectly monotonic in area, probably because included districts/files differ by dataset.
- SHP geometries are simplified before sending to Earth Engine to avoid payload limits.

## Reliability Statement

Recommended wording:

> The selected GTB hybrid model provides a credible static flood susceptibility ranking for Seoul. Spatial 5-fold validation shows stable classification performance, and top-k validation indicates that high-risk areas capture held-out flood reference points far more efficiently than random area selection. External comparison with official urban flood maps also supports spatial alignment between predicted high-risk zones and official flood-prone areas. However, the model should be interpreted as a static susceptibility model, not a rainfall-event or real-time inundation forecast.

## Remaining Work

1. Run buffer sensitivity across all 5 folds.
2. Add city-holdout validation when another city's labels/features are ready.
3. Add probability calibration checks if the output will be interpreted as probability rather than rank.
4. Consider positive-unlabeled framing or stronger true-negative data.
5. Keep official map validation as external support, but do not treat it as event-date truth.
