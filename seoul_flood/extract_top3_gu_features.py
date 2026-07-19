"""Extract paper-ready feature statistics for Seoul's top flood-risk districts.

The script trains only the final all-Seoul model from ``run_analysis.py`` (it
does not repeat the five-fold validation) and summarizes every model input
feature over a comparison design that supports the paper's spatial
interpretation section:

Regions
    * ``seoul_all`` - all of Seoul (baseline)
    * ``district``  - every one of Seoul's 25 자치구, reported separately

Districts are never merged into an aggregate bucket, so any district can be
used as a comparison partner when the paper is written.  The ``--districts``
option only marks which districts are the paper's focus (default is the top 3
of the ranking CSV); every district is measured regardless.

Pixel classes (within every region)
    * ``all``     - every valid 30 m analysis pixel
    * ``hotspot`` - pixels at or above the Seoul-wide flood probability
      percentile (default p95), i.e. the model's high-risk zone
    * ``normal``  - the remaining pixels

Run from the repository root::

    python seoul_flood/extract_top3_gu_features.py

Use explicit focus districts instead of the ranking CSV if necessary::

    python seoul_flood/extract_top3_gu_features.py \
        --districts 강남구 강동구 동작구
"""

import argparse
import csv
import json
import os

import ee


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUTPUT_DIR = os.path.join(SCRIPT_DIR, "outputs", "analysis")
DEFAULT_RANKING_CSV = os.path.join(
    DEFAULT_OUTPUT_DIR,
    "top5_red_points_by_gu.csv",
)
DEFAULT_DISTRICTS = ["강남구", "강동구", "동작구"]

SEOUL_ALL_LABEL = "서울전체"

# Region ordering used for every output file: the Seoul baseline first, then
# the 25 districts by flood-risk rank.
SEOUL_ALL_ORDER = 0
DISTRICT_ORDER_BASE = 10

PIXEL_CLASS_ORDER = {"all": 0, "hotspot": 1, "normal": 2}


# Unit and interpretation notes are saved beside the numeric result so the
# values can be described accurately when the paper table is written.
FEATURE_METADATA = {
    "slope": {
        "label_ko": "경사도",
        "unit": "degree",
        "description_ko": "SRTM 수치표고모델에서 계산한 지표면 경사도",
    },
    "hnd": {
        "label_ko": "최근 배수로 대비 높이",
        "unit": "m",
        "description_ko": "MERIT Hydro Height Above Nearest Drainage",
    },
    "log_upa": {
        "label_ko": "로그 상류 집수면적",
        "unit": "ln(km2 + 1)",
        "description_ko": "MERIT Hydro 상류 집수면적에 1을 더한 자연로그",
    },
    "water_occ": {
        "label_ko": "표면수 발생 빈도",
        "unit": "ratio (0-1)",
        "description_ko": "JRC Global Surface Water occurrence를 0~1로 변환한 값",
    },
    "built": {
        "label_ko": "건조 환경 확률",
        "unit": "probability (0-1)",
        "description_ko": "분석 연도 Dynamic World built class의 연평균 확률",
    },
    "lowland": {
        "label_ko": "저지대 지수",
        "unit": "normalized score (0-1)",
        "description_ko": "서울 전체 고도 범위에서 낮은 정도; 1에 가까울수록 저지대",
    },
    "alpha_score": {
        "label_ko": "AlphaEarth 침수 유사도",
        "unit": "normalized score (0-1)",
        "description_ko": "과거 침수 기준점 평균 embedding과의 유사도",
    },
    "pump_capacity_sum_norm": {
        "label_ko": "배수펌프장 최대배수량 합계(정규화)",
        "unit": "normalized score (0-1)",
        "description_ko": "자치구별 배수펌프장 최대배수량 합계를 서울 최댓값으로 나눈 값",
    },
    "pump_catchment_area_sum_norm": {
        "label_ko": "배수펌프장 유역면적 합계(정규화)",
        "unit": "normalized score (0-1)",
        "description_ko": "자치구별 배수펌프장 유역면적 합계를 서울 최댓값으로 나눈 값",
    },
    "pump_reservoir_capacity_sum_norm": {
        "label_ko": "배수펌프장 유수지 용량 합계(정규화)",
        "unit": "normalized score (0-1)",
        "description_ko": "자치구별 배수펌프장 유수지 용량 합계를 서울 최댓값으로 나눈 값",
    },
    "pump_station_count_norm": {
        "label_ko": "배수펌프장 수(정규화)",
        "unit": "normalized score (0-1)",
        "description_ko": "자치구별 배수펌프장 수를 서울 최댓값으로 나눈 값",
    },
    "sewer_sensor_count_norm": {
        "label_ko": "하수관로 수위센서 수(정규화)",
        "unit": "normalized score (0-1)",
        "description_ko": "자치구별 하수관로 수위센서 수를 서울 최댓값으로 나눈 값",
    },
    "flood_prob": {
        "label_ko": "모델 침수 위험도",
        "unit": "probability (0-1)",
        "description_ko": "최종 GTB soft-voting 모델의 침수 확률 출력",
    },
}


# 순위표를 읽을 때 "값이 크면 위험"인지 "작으면 위험"인지 알려주는 힌트다.
# 통상적인 수문학적 해석이며 모델이 학습한 방향이 아니다.  배수 인프라는 침수가
# 잦은 곳에 더 많이 설치되기도 해서 방향을 단정할 수 없으므로 비워 둔다.
RISK_DIRECTION = {
    "slope": "lower",
    "hnd": "lower",
    "log_upa": "higher",
    "water_occ": "higher",
    "built": "higher",
    "lowland": "higher",
    "alpha_score": "higher",
    "flood_prob": "higher",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="서울시 침수 위험도 상위 자치구의 모델 Feature 기술통계와 비교표를 추출합니다."
    )
    parser.add_argument(
        "--ranking-csv",
        default=DEFAULT_RANKING_CSV,
        help="create_final_flood_risk_map.py가 생성한 구별 순위 CSV",
    )
    parser.add_argument(
        "--districts",
        nargs="+",
        help="순위 CSV 대신 사용할 자치구명(예: 강남구 강동구 동작구)",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=3,
        help="순위 CSV에서 선택할 상위 자치구 수(기본값: 3)",
    )
    parser.add_argument(
        "--hotspot-percentile",
        type=int,
        default=95,
        help="고위험 pixel 기준이 되는 서울 전체 침수 확률 백분위(기본값: 95)",
    )
    parser.add_argument(
        "--tile-scale",
        type=int,
        default=8,
        help="Earth Engine reduceRegions tileScale(메모리 부족 시 값을 키우세요)",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="결과 CSV/JSON 저장 폴더",
    )
    return parser.parse_args()


def read_ranking_rows(path):
    if not os.path.exists(path):
        return []

    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    parsed = []
    for row in rows:
        gu_name = (row.get("gu_name") or "").strip()
        if not gu_name:
            continue
        try:
            rank = int(row.get("rank") or 0)
        except ValueError:
            rank = 0
        parsed.append(
            {
                "gu_name": gu_name,
                "rank": rank,
                "red_point_count": _optional_number(row.get("red_point_count"), int),
                "red_area_km2": _optional_number(row.get("red_area_km2"), float),
            }
        )
    return sorted(parsed, key=lambda row: (row["rank"] or 10**9, row["gu_name"]))


def _optional_number(value, number_type):
    if value is None or str(value).strip() == "":
        return None
    try:
        return number_type(float(value))
    except (TypeError, ValueError):
        return None


def select_focus_districts(args, ranking_rows):
    """Pick the districts the paper focuses on.

    Every district is measured either way; this only decides which ones are
    flagged in the outputs and listed first in the contrast table.
    """
    if args.districts:
        names = args.districts
        selection_source = "command_line"
    elif ranking_rows:
        names = [row["gu_name"] for row in ranking_rows[: args.top_n]]
        selection_source = os.path.abspath(args.ranking_csv)
    else:
        names = DEFAULT_DISTRICTS[: args.top_n]
        selection_source = "default_fallback"

    focus = []
    for gu_name in names:
        if gu_name not in focus:
            focus.append(gu_name)
    return focus, selection_source


def build_all_district_rows(analysis, ranking_rows, focus_names):
    """Describe all 25 자치구, ordered by flood-risk rank where available."""
    unknown = [name for name in focus_names if name not in analysis.SEOUL_GU_NAMES]
    if unknown:
        raise ValueError(f"서울 자치구명을 확인하세요: {', '.join(unknown)}")

    ranking_by_gu = {row["gu_name"]: row for row in ranking_rows}
    rows = []
    for gu_name in analysis.SEOUL_GU_ORDER:
        ranking = ranking_by_gu.get(gu_name, {})
        rows.append(
            {
                "gu_name": gu_name,
                "rank": ranking.get("rank"),
                "red_point_count": ranking.get("red_point_count"),
                "red_area_km2": ranking.get("red_area_km2"),
                "is_focus_district": gu_name in focus_names,
            }
        )
    # Districts missing from the ranking CSV keep a stable position at the end
    # instead of colliding on a single sort key.
    rows.sort(
        key=lambda row: (
            row["rank"] if row["rank"] else 10**6 + analysis.SEOUL_GU_ORDER.index(row["gu_name"]),
        )
    )
    for index, row in enumerate(rows):
        row["region_order"] = DISTRICT_ORDER_BASE + index
    return rows


def build_region_collection(analysis, district_rows):
    """All 25 district polygons plus the Seoul-wide baseline, as one collection."""
    # Earth Engine dictionaries cannot contain Python ``None`` values.  Only
    # properties needed on the server are sent; CSV-only metadata stays local.
    metadata_by_shape = {
        analysis.SEOUL_GU_TO_ADM2_SHAPE[row["gu_name"]]: {
            "region_name": row["gu_name"],
            "region_order": row["region_order"],
        }
        for row in district_rows
    }
    metadata = ee.Dictionary(metadata_by_shape)
    shape_names = list(metadata_by_shape)
    districts = (
        ee.FeatureCollection("WM/geoLab/geoBoundaries/600/ADM2")
        .filter(ee.Filter.eq("shapeGroup", "KOR"))
        .filterBounds(analysis.seoul)
        .filter(ee.Filter.inList("shapeName", shape_names))
    )

    def _set_metadata(feature):
        values = ee.Dictionary(metadata.get(feature.get("shapeName")))
        return feature.set(
            {
                "region_name": values.get("region_name"),
                "region_type": "district",
                "region_order": values.get("region_order"),
            }
        )

    districts = districts.map(_set_metadata)
    matched = districts.size().getInfo()
    if matched != len(district_rows):
        raise ValueError(
            f"ADM2 경계 매칭 실패: 요청 {len(district_rows)}개, 매칭 {matched}개"
        )

    seoul_feature = ee.Feature(
        analysis.seoul,
        {
            "region_name": SEOUL_ALL_LABEL,
            "region_type": "seoul_all",
            "region_order": SEOUL_ALL_ORDER,
        },
    )
    return districts.merge(ee.FeatureCollection([seoul_feature]))


def build_statistics_reducer():
    return (
        ee.Reducer.mean()
        .combine(ee.Reducer.median(), "", True)
        .combine(ee.Reducer.stdDev(), "", True)
        .combine(ee.Reducer.minMax(), "", True)
        .combine(ee.Reducer.percentile([5, 25, 75, 95]), "", True)
        .combine(ee.Reducer.count(), "", True)
    )


def reduce_pixel_class(image, regions, analysis_scale, tile_scale, pixel_class):
    features = image.reduceRegions(
        collection=regions,
        reducer=build_statistics_reducer(),
        scale=analysis_scale,
        tileScale=tile_scale,
    ).getInfo()["features"]
    return [(pixel_class, feature["properties"]) for feature in features]


def metadata_for_feature(feature):
    if feature in FEATURE_METADATA:
        return FEATURE_METADATA[feature]
    if feature.endswith("_norm"):
        return {
            "label_ko": feature,
            "unit": "normalized score (0-1)",
            "description_ko": "자치구별 배수 인프라 원자료를 서울 최댓값 기준으로 정규화한 모델 입력값",
        }
    return {
        "label_ko": feature,
        "unit": "",
        "description_ko": "",
    }


STAT_SUFFIXES = {
    "mean": "mean",
    "median": "median",
    "std_dev": "stdDev",
    "min": "min",
    "p05": "p5",
    "p25": "p25",
    "p75": "p75",
    "p95": "p95",
    "max": "max",
    "valid_pixel_count": "count",
}


def make_statistics_rows(
    class_properties,
    feature_names,
    constant_features,
    focus_names,
):
    """Build the long statistics table plus a lookup used by the contrast table.

    ``stats_index`` maps ``(region_name, pixel_class, feature)`` to that cell's
    statistics so district/baseline pairs can be differenced without another
    Earth Engine round trip.
    """
    long_rows = []
    mean_rows_by_key = {}
    stats_index = {}
    region_info = {}

    for pixel_class, props in class_properties:
        region_name = props["region_name"]
        region_type = props["region_type"]
        region_order = props["region_order"]
        is_focus = region_name in focus_names
        region_info[region_name] = {
            "region_type": region_type,
            "region_order": region_order,
            "is_focus_district": is_focus,
        }
        mean_row = {
            "region_order": region_order,
            "region_name": region_name,
            "region_type": region_type,
            "is_focus_district": is_focus,
            "pixel_class": pixel_class,
        }
        for feature in feature_names:
            feature_meta = metadata_for_feature(feature)
            stats = {
                output_name: props.get(f"{feature}_{ee_suffix}")
                for output_name, ee_suffix in STAT_SUFFIXES.items()
            }
            stats_index[(region_name, pixel_class, feature)] = stats
            long_rows.append(
                {
                    "region_order": region_order,
                    "region_name": region_name,
                    "region_type": region_type,
                    "is_focus_district": is_focus,
                    "pixel_class": pixel_class,
                    "feature": feature,
                    "variable_role": (
                        "model_output" if feature == "flood_prob" else "model_input"
                    ),
                    "feature_label_ko": feature_meta["label_ko"],
                    "unit": feature_meta["unit"],
                    "is_district_constant": feature in constant_features,
                    **stats,
                    "description_ko": feature_meta["description_ko"],
                }
            )
            mean_row[feature] = stats["mean"]
        mean_rows_by_key[(region_order, pixel_class)] = mean_row

    long_rows.sort(
        key=lambda row: (
            row["region_order"],
            PIXEL_CLASS_ORDER.get(row["pixel_class"], 99),
            row["feature"],
        )
    )
    mean_rows = [
        mean_rows_by_key[key]
        for key in sorted(
            mean_rows_by_key,
            key=lambda key: (key[0], PIXEL_CLASS_ORDER.get(key[1], 99)),
        )
    ]
    return long_rows, mean_rows, stats_index, region_info


def cohens_d(stats_a, stats_b):
    """Standardized mean difference using the pooled standard deviation.

    Neighbouring 30 m pixels are strongly autocorrelated, so pixel counts are
    not independent samples and p-values would be meaningless here.  The effect
    size is reported instead as a scale-free measure of how far apart two
    groups sit.
    """
    mean_a, mean_b = stats_a.get("mean"), stats_b.get("mean")
    std_a, std_b = stats_a.get("std_dev"), stats_b.get("std_dev")
    count_a, count_b = stats_a.get("valid_pixel_count"), stats_b.get("valid_pixel_count")
    if None in (mean_a, mean_b, std_a, std_b, count_a, count_b):
        return None
    if count_a < 2 or count_b < 2:
        return None
    pooled_variance = (
        (count_a - 1) * std_a**2 + (count_b - 1) * std_b**2
    ) / (count_a + count_b - 2)
    if pooled_variance <= 0:
        return None
    return (mean_a - mean_b) / pooled_variance**0.5


def make_contrast_rows(stats_index, district_rows, feature_names, constant_features):
    """Pair every district against the baselines the paper argues from.

    Contrasts are produced for all 25 districts, with the focus districts first,
    so any district can serve as a comparison partner in the write-up.
    """
    ordered = sorted(
        district_rows,
        key=lambda row: (not row["is_focus_district"], row["region_order"]),
    )

    pairs = []
    for row in ordered:
        gu_name = row["gu_name"]
        # 이 구가 서울 평균과 어떻게 다른가.
        pairs.append(
            (
                "district_vs_seoul",
                f"{gu_name} 전체 격자 vs 서울 전체 격자",
                gu_name,
                (gu_name, "all"),
                (SEOUL_ALL_LABEL, "all"),
            )
        )
        # 같은 구 안에서 위험 격자와 일반 격자를 가르는 요인.
        pairs.append(
            (
                "hotspot_vs_normal_within_district",
                f"{gu_name} 위험 격자 vs {gu_name} 일반 격자",
                gu_name,
                (gu_name, "hotspot"),
                (gu_name, "normal"),
            )
        )
        # 이 구의 위험 격자가 서울 전체 위험 격자와 다른, 지역 고유의 특성.
        pairs.append(
            (
                "district_hotspot_vs_seoul_hotspot",
                f"{gu_name} 위험 격자 vs 서울 전체 위험 격자",
                gu_name,
                (gu_name, "hotspot"),
                (SEOUL_ALL_LABEL, "hotspot"),
            )
        )

    # 서울 전체 기준의 위험/일반 대비: 개별 구 결과를 읽을 때의 기준선.
    pairs.append(
        (
            "hotspot_vs_normal_seoul",
            "서울 전체 위험 격자 vs 서울 전체 일반 격자",
            SEOUL_ALL_LABEL,
            (SEOUL_ALL_LABEL, "hotspot"),
            (SEOUL_ALL_LABEL, "normal"),
        )
    )

    focus_names = {row["gu_name"] for row in district_rows if row["is_focus_district"]}
    rows = []
    for order, (comparison, label, region_name, key_a, key_b) in enumerate(pairs):
        for feature in feature_names:
            stats_a = stats_index.get((key_a[0], key_a[1], feature))
            stats_b = stats_index.get((key_b[0], key_b[1], feature))
            if stats_a is None or stats_b is None:
                continue
            feature_meta = metadata_for_feature(feature)
            mean_a, mean_b = stats_a["mean"], stats_b["mean"]
            mean_diff = None if None in (mean_a, mean_b) else mean_a - mean_b
            ratio = None
            if mean_a is not None and mean_b:
                ratio = mean_a / mean_b
            rows.append(
                {
                    "comparison_order": order,
                    "comparison_type": comparison,
                    "comparison_label_ko": label,
                    "region_name": region_name,
                    "is_focus_district": region_name in focus_names,
                    "feature": feature,
                    "variable_role": (
                        "model_output" if feature == "flood_prob" else "model_input"
                    ),
                    "feature_label_ko": feature_meta["label_ko"],
                    "unit": feature_meta["unit"],
                    "is_district_constant": feature in constant_features,
                    "group_a": f"{key_a[0]}/{key_a[1]}",
                    "group_a_mean": mean_a,
                    "group_a_std_dev": stats_a["std_dev"],
                    "group_a_pixel_count": stats_a["valid_pixel_count"],
                    "group_b": f"{key_b[0]}/{key_b[1]}",
                    "group_b_mean": mean_b,
                    "group_b_std_dev": stats_b["std_dev"],
                    "group_b_pixel_count": stats_b["valid_pixel_count"],
                    "mean_diff": mean_diff,
                    "mean_ratio": ratio,
                    "cohens_d": cohens_d(stats_a, stats_b),
                }
            )
    return rows


def make_ranking_rows(stats_index, district_rows, feature_names):
    """Rank the 25 districts against each other on every feature.

    This is what replaces the old aggregate '기타자치구' bucket: instead of one
    lumped baseline, each district gets its position within the 25-district
    distribution, so the paper can say exactly where a district stands.
    """
    focus_names = {row["gu_name"] for row in district_rows if row["is_focus_district"]}
    rows = []
    for pixel_class in sorted(PIXEL_CLASS_ORDER, key=PIXEL_CLASS_ORDER.get):
        for feature in feature_names:
            feature_meta = metadata_for_feature(feature)
            measured = []
            for row in district_rows:
                stats = stats_index.get((row["gu_name"], pixel_class, feature))
                if stats is None or stats.get("mean") is None:
                    continue
                measured.append((row["gu_name"], stats["mean"]))
            if not measured:
                continue

            values = [mean for _, mean in measured]
            district_mean = sum(values) / len(values)
            spread = (
                sum((value - district_mean) ** 2 for value in values)
                / (len(values) - 1)
            ) ** 0.5 if len(values) > 1 else 0.0

            # 값이 큰 구가 1위. 방향 해석은 risk_direction 컬럼을 함께 본다.
            measured.sort(key=lambda item: -item[1])
            for rank, (gu_name, mean_value) in enumerate(measured, start=1):
                rows.append(
                    {
                        "pixel_class": pixel_class,
                        "feature": feature,
                        "feature_label_ko": feature_meta["label_ko"],
                        "unit": feature_meta["unit"],
                        "risk_direction": RISK_DIRECTION.get(feature, ""),
                        "gu_name": gu_name,
                        "is_focus_district": gu_name in focus_names,
                        "mean": mean_value,
                        "rank_desc": rank,
                        "district_count": len(measured),
                        "district_mean": district_mean,
                        "district_std_dev": spread,
                        "z_vs_districts": (
                            (mean_value - district_mean) / spread if spread else None
                        ),
                    }
                )
    return rows


def write_csv(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def resolve_hotspot_threshold(probability, analysis, percentile):
    threshold_info = probability.reduceRegion(
        reducer=ee.Reducer.percentile([percentile]),
        geometry=analysis.seoul,
        scale=analysis.ANALYSIS_SCALE,
        maxPixels=1e9,
        tileScale=4,
    ).getInfo()
    threshold = threshold_info.get(f"flood_prob_p{percentile}")
    # Depending on the Earth Engine API version, a one-percentile reducer can
    # return either ``flood_prob_p95`` or the unsuffixed ``flood_prob`` key.
    if threshold is None:
        threshold = threshold_info.get("flood_prob")
    if threshold is None and len(threshold_info) == 1:
        threshold = next(iter(threshold_info.values()))
    if threshold is None:
        raise ValueError(f"고위험 임계값을 읽지 못했습니다: {threshold_info}")
    return threshold


def main():
    args = parse_args()
    if args.top_n < 1:
        raise ValueError("--top-n은 1 이상이어야 합니다.")
    if not 0 < args.hotspot_percentile < 100:
        raise ValueError("--hotspot-percentile은 0과 100 사이여야 합니다.")
    if args.tile_scale < 1:
        raise ValueError("--tile-scale은 1 이상이어야 합니다.")

    ranking_rows = read_ranking_rows(args.ranking_csv)
    focus_names, selection_source = select_focus_districts(args, ranking_rows)

    # Importing run_analysis initializes Earth Engine and builds the exact same
    # base/AlphaEarth inputs used by the final flood-risk map.
    try:
        from . import run_analysis as analysis
    except ImportError:
        import run_analysis as analysis

    district_rows = build_all_district_rows(analysis, ranking_rows, focus_names)
    regions = build_region_collection(analysis, district_rows)
    print(f"측정 대상: 서울 {len(district_rows)}개 자치구 전체 + 서울 전체 기준선")
    print("논문 focus 자치구:", ", ".join(focus_names))
    print("최종 전체-서울 모델을 학습합니다(5-fold 교차검증은 생략).")
    final_model = analysis.run_final_model()

    probability = final_model["probability"].rename("flood_prob")
    feature_names = list(analysis.INPUT_BANDS) + ["flood_prob"]
    # 배수 인프라 feature는 자치구 단위 상수라 구 내부(위험/일반) 대비에서는
    # 항상 0이 나온다. 표를 읽을 때 혼동하지 않도록 플래그로 표시한다.
    constant_features = set(analysis.static_features["drainage_bands"])
    result_image = final_model["feature_image"].select(
        analysis.INPUT_BANDS
    ).addBands(probability)

    hotspot_threshold = resolve_hotspot_threshold(
        probability,
        analysis,
        args.hotspot_percentile,
    )
    print(f"고위험 임계 확률(p{args.hotspot_percentile}): {hotspot_threshold}")
    hotspot_mask = probability.gte(ee.Number(hotspot_threshold))

    images_by_class = {
        "all": result_image,
        "hotspot": result_image.updateMask(hotspot_mask),
        "normal": result_image.updateMask(hotspot_mask.Not()),
    }

    class_properties = []
    for pixel_class, image in images_by_class.items():
        print(f"'{pixel_class}' 격자 통계를 계산합니다...")
        class_properties.extend(
            reduce_pixel_class(
                image,
                regions,
                analysis.ANALYSIS_SCALE,
                args.tile_scale,
                pixel_class,
            )
        )

    long_rows, mean_rows, stats_index, region_info = make_statistics_rows(
        class_properties,
        feature_names,
        constant_features,
        set(focus_names),
    )
    contrast_rows = make_contrast_rows(
        stats_index,
        district_rows,
        feature_names,
        constant_features,
    )
    ranking_out_rows = make_ranking_rows(stats_index, district_rows, feature_names)

    output_dir = os.path.abspath(args.output_dir)
    statistics_csv = os.path.join(output_dir, "seoul_gu_feature_statistics.csv")
    means_csv = os.path.join(output_dir, "seoul_gu_feature_means.csv")
    contrast_csv = os.path.join(output_dir, "seoul_gu_feature_contrast.csv")
    ranking_csv = os.path.join(output_dir, "seoul_gu_feature_ranking.csv")
    metadata_json = os.path.join(output_dir, "seoul_gu_feature_metadata.json")
    write_csv(statistics_csv, long_rows)
    write_csv(means_csv, mean_rows)
    write_csv(contrast_csv, contrast_rows)
    write_csv(ranking_csv, ranking_out_rows)
    write_json(
        metadata_json,
        {
            "year": analysis.YEAR,
            "analysis_scale_m": analysis.ANALYSIS_SCALE,
            "focus_district_selection_source": selection_source,
            "focus_districts": focus_names,
            "districts": district_rows,
            "regions": region_info,
            "input_features": list(analysis.INPUT_BANDS),
            "model_output_included": "flood_prob",
            "district_constant_features": sorted(constant_features),
            "hotspot_percentile": args.hotspot_percentile,
            "hotspot_probability_threshold": hotspot_threshold,
            "pixel_classes": {
                "all": "해당 지역의 전체 유효 pixel",
                "hotspot": (
                    f"침수 확률이 서울 전체 p{args.hotspot_percentile} 이상인 위험 pixel"
                ),
                "normal": (
                    f"침수 확률이 서울 전체 p{args.hotspot_percentile} 미만인 일반 pixel"
                ),
            },
            "region_types": {
                "seoul_all": "서울시 전체",
                "district": "서울시 25개 자치구 각각(합산 없음)",
            },
            "comparison_types": {
                "district_vs_seoul": "자치구 전체 격자 - 서울 전체 격자",
                "hotspot_vs_normal_within_district": (
                    "동일 자치구 내 위험 격자 - 일반 격자"
                ),
                "district_hotspot_vs_seoul_hotspot": (
                    "자치구 위험 격자 - 서울 전체 위험 격자"
                ),
                "hotspot_vs_normal_seoul": "서울 전체 위험 격자 - 일반 격자",
            },
            "ranking_note_ko": (
                "seoul_gu_feature_ranking.csv는 feature별로 25개 자치구를 값이 "
                "큰 순서(rank_desc=1이 최대)로 정렬한다. risk_direction은 값이 "
                "크고 작음 중 어느 쪽이 침수 위험을 뜻하는지에 대한 통상적 "
                "수문학 해석이며 모델이 학습한 방향은 아니다. 배수 인프라는 "
                "침수가 잦은 곳에 더 설치되기도 해 방향을 비워 두었다."
            ),
            "statistics": list(STAT_SUFFIXES),
            "effect_size_note_ko": (
                "cohens_d는 pooled 표준편차로 표준화한 평균 차이다. 인접 30 m "
                "격자는 공간적으로 강하게 자기상관되어 pixel 수를 독립 표본 "
                "크기로 볼 수 없으므로 유의성 검정 대신 효과크기만 보고한다."
            ),
            "feature_metadata": {
                feature: metadata_for_feature(feature) for feature in feature_names
            },
            "outputs": {
                "feature_statistics_csv": statistics_csv,
                "feature_means_csv": means_csv,
                "feature_contrast_csv": contrast_csv,
                "feature_ranking_csv": ranking_csv,
                "metadata_json": metadata_json,
            },
        },
    )

    print("저장 완료:")
    for path in (statistics_csv, means_csv, contrast_csv, ranking_csv, metadata_json):
        print(f"- {path}")
    return {
        "feature_statistics_csv": statistics_csv,
        "feature_means_csv": means_csv,
        "feature_contrast_csv": contrast_csv,
        "feature_ranking_csv": ranking_csv,
        "metadata_json": metadata_json,
    }


if __name__ == "__main__":
    main()
