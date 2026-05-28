import json
import os

import ee
import geemap


def load_env_file(path=".env"):
    """간단한 .env 파일을 읽어 환경변수로 등록한다."""
    if not os.path.exists(path):
        return

    with open(path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue

            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip("\"").strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


load_env_file()

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def resolve_input_path(path, fallback_dir=None):
    """상대 입력 경로를 현재 실행 위치와 스크립트 위치 기준으로 안전하게 해석한다."""
    if os.path.isabs(path):
        return path

    candidates = [
        os.path.abspath(path),
        os.path.join(SCRIPT_DIR, path),
    ]
    if fallback_dir:
        candidates.append(os.path.join(fallback_dir, os.path.basename(path)))

    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return candidates[-1]


def resolve_output_path(path):
    """상대 출력 경로는 이 분석 스크립트 폴더 아래에 저장한다."""
    if os.path.isabs(path):
        return path
    return os.path.join(SCRIPT_DIR, path)


def parse_int_list(raw_value):
    """쉼표로 구분된 정수 환경변수 값을 리스트로 변환한다."""
    return [
        int(value.strip())
        for value in raw_value.split(",")
        if value.strip()
    ]


# 실행 환경에서 조정할 수 있는 주요 설정값들이다.
# EE_PROJECT_ID만 필수이고, 나머지는 없으면 기본값을 사용한다.
PROJECT_ID = os.environ.get("EE_PROJECT_ID")
if not PROJECT_ID:
    raise ValueError("EE_PROJECT_ID is missing. Set it in .env or your shell environment.")

YEAR = int(os.environ.get("YEAR", "2024"))
OUTPUT_HTML = resolve_output_path(os.environ.get("OUTPUT_HTML", "seoul_flood_risk_rf.html"))
ANALYSIS_SCALE = int(os.environ.get("ANALYSIS_SCALE", "30"))
SEOUL_REFERENCE_GEOJSON = resolve_input_path(
    os.environ.get("SEOUL_REFERENCE_GEOJSON", "source/seoul_flood_reference_points.geojson"),
    fallback_dir=os.path.join(SCRIPT_DIR, "source"),
)
REFERENCE_POINT_LIMIT = int(os.environ.get("REFERENCE_POINT_LIMIT", "0"))
POSITIVE_SAMPLE_POINTS = int(
    os.environ.get("POSITIVE_SAMPLE_POINTS", os.environ.get("POSITIVE_POINTS", "200"))
)
NEGATIVE_POINTS = int(os.environ.get("NEGATIVE_POINTS", "200"))
NEGATIVE_BUFFER_M = int(os.environ.get("NEGATIVE_BUFFER_M", "300"))
POSITIVE_BUFFER_M = int(os.environ.get("POSITIVE_BUFFER_M", "60"))
HOTSPOT_PERCENTILE = int(os.environ.get("HOTSPOT_PERCENTILE", "95"))
SPATIAL_BLOCK_DEGREES = float(os.environ.get("SPATIAL_BLOCK_DEGREES", "0.015"))
SPATIAL_FOLDS = int(os.environ.get("SPATIAL_FOLDS", "5"))
VALIDATION_FOLD = int(os.environ.get("VALIDATION_FOLD", "0"))
RUN_BUFFER_SENSITIVITY = os.environ.get("RUN_BUFFER_SENSITIVITY", "1") == "1"
POSITIVE_BUFFER_SWEEP_M = parse_int_list(
    os.environ.get("POSITIVE_BUFFER_SWEEP_M", "30,60,90")
)
NEGATIVE_BUFFER_SWEEP_M = parse_int_list(
    os.environ.get("NEGATIVE_BUFFER_SWEEP_M", "200,300,500")
)
BUFFER_SENSITIVITY_FOLDS = parse_int_list(
    os.environ.get("BUFFER_SENSITIVITY_FOLDS", str(VALIDATION_FOLD))
)

# Google Earth Engine에 접속한다.
ee.Initialize(project=PROJECT_ID)


def add_spatial_fold(fc):
    """위도/경도 격자 블록을 기준으로 공간 검증 fold를 붙인다."""

    def _add_fold(feature):
        coords = feature.geometry().coordinates()
        lon = ee.Number(coords.get(0))
        lat = ee.Number(coords.get(1))
        block_x = lon.add(180).divide(SPATIAL_BLOCK_DEGREES).floor()
        block_y = lat.add(90).divide(SPATIAL_BLOCK_DEGREES).floor()
        fold = (
            block_x.multiply(73856093)
            .add(block_y.multiply(19349663))
            .mod(SPATIAL_FOLDS)
            .int()
        )
        return feature.set({"block_x": block_x, "block_y": block_y, "fold": fold})

    return fc.map(_add_fold)


# -------------------------------------------------
# Seoul boundary
# -------------------------------------------------
# geoBoundaries의 한국 ADM1 행정구역 중 서울 경계만 골라 분석 영역으로 사용한다.
adm1 = ee.FeatureCollection("WM/geoLab/geoBoundaries/600/ADM1")
kor_adm1 = adm1.filter(ee.Filter.eq("shapeGroup", "KOR"))
seoul_fc = kor_adm1.filter(
    ee.Filter.Or(
        ee.Filter.stringContains("shapeName", "Seoul"),
        ee.Filter.stringContains("shapeName", "SEOUL"),
        ee.Filter.stringContains("shapeName", "서울"),
    )
)
seoul_count = seoul_fc.size().getInfo()
print("Matched Seoul features:", seoul_count)
if seoul_count == 0:
    raise ValueError("서울 경계를 찾지 못했습니다.")
seoul = seoul_fc.geometry()

# 서울을 일정한 위경도 격자로 나누고, 일부 격자 fold 전체를 검증 영역으로 남긴다.
lonlat = ee.Image.pixelLonLat()
block_x_img = lonlat.select("longitude").add(180).divide(SPATIAL_BLOCK_DEGREES).floor()
block_y_img = lonlat.select("latitude").add(90).divide(SPATIAL_BLOCK_DEGREES).floor()
spatial_fold = (
    block_x_img.multiply(73856093)
    .add(block_y_img.multiply(19349663))
    .mod(SPATIAL_FOLDS)
    .rename("spatial_fold")
    .clip(seoul)
)
train_area_mask = spatial_fold.neq(VALIDATION_FOLD)
valid_area_mask = spatial_fold.eq(VALIDATION_FOLD)
print(
    "Spatial validation:",
    {
        "block_degrees": SPATIAL_BLOCK_DEGREES,
        "folds": SPATIAL_FOLDS,
        "validation_fold": VALIDATION_FOLD,
    },
)

# 서울시 침수 흔적/기준점 GeoJSON을 양성(label=1) 학습 데이터로 사용한다.
with open(SEOUL_REFERENCE_GEOJSON, "r", encoding="utf-8") as f:
    seoul_reference_geojson = json.load(f)

reference_features = seoul_reference_geojson["features"]
if REFERENCE_POINT_LIMIT > 0:
    reference_features = reference_features[:REFERENCE_POINT_LIMIT]

positive_features = [
    ee.Feature(
        ee.Geometry.Point(feature["geometry"]["coordinates"]),
        {
            **feature["properties"],
            "label": 1,
        },
    )
    for feature in reference_features
]
positive_points = add_spatial_fold(ee.FeatureCollection(positive_features))
positive_geom = positive_points.geometry()
train_positive_points = positive_points.filter(ee.Filter.neq("fold", VALIDATION_FOLD))
valid_positive_points = positive_points.filter(ee.Filter.eq("fold", VALIDATION_FOLD))
train_positive_geom = train_positive_points.geometry()
train_positive_count = train_positive_points.size().getInfo()
valid_positive_count = valid_positive_points.size().getInfo()
positive_fold_hist = positive_points.aggregate_histogram("fold").getInfo()
print("Positive flood points:", len(positive_features))
print("Positive points by spatial fold:", positive_fold_hist)
print("Train / validation positive reference points:", train_positive_count, valid_positive_count)
print("Sample points per class:", {"positive": POSITIVE_SAMPLE_POINTS, "negative": NEGATIVE_POINTS})
if train_positive_count == 0 or valid_positive_count == 0:
    raise ValueError(
        "공간 검증 fold에 양성점이 부족합니다. VALIDATION_FOLD 또는 SPATIAL_BLOCK_DEGREES를 조정하세요."
    )

# -------------------------------------------------
# Feature stack for Seoul
# -------------------------------------------------
# 침수 가능성과 관련된 설명변수(feature)를 Earth Engine 영상으로 만든다.
# 각 픽셀은 경사, 하천 주변 지형, 물/도시화 정도, 저지대 정도, AlphaEarth 유사도 값을 갖게 된다.
dem = ee.Image("USGS/SRTMGL1_003").clip(seoul)
slope = ee.Terrain.slope(dem).rename("slope")
merit = ee.Image("MERIT/Hydro/v1_0_1").clip(seoul)
hnd = merit.select("hnd").rename("hnd")
log_upa = merit.select("upa").add(1).log().rename("log_upa")

# 장기간 물이 관측된 곳은 기존 수역일 가능성이 높으므로 침수 판단의 보조 특징으로 사용한다.
water_occ = (
    ee.Image("JRC/GSW1_4/GlobalSurfaceWater")
    .select("occurrence")
    .clip(seoul)
    .unmask(0)
    .divide(100)
    .rename("water_occ")
)

# Dynamic World의 연평균 built/water 확률로 도시화 정도와 최근 물 영역을 반영한다.
dw = (
    ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1")
    .filterDate(f"{YEAR}-01-01", f"{YEAR+1}-01-01")
    .filterBounds(seoul)
    .select(["built", "water"])
    .mean()
    .clip(seoul)
)
built = dw.select("built").rename("built")
dw_water = dw.select("water").rename("dw_water")

# 서울 내부의 상대 고도값을 0~1로 정규화한다.
# 값이 클수록 서울 안에서 상대적으로 낮은 저지대라는 뜻이다.
dem_stats = dem.reduceRegion(
    reducer=ee.Reducer.minMax(),
    geometry=seoul,
    scale=ANALYSIS_SCALE,
    maxPixels=1e9,
).getInfo()
dem_min = dem_stats["elevation_min"]
dem_max = dem_stats["elevation_max"]
dem_range = max(dem_max - dem_min, 1e-6)
lowland = (
    ee.Image.constant(dem_max)
    .subtract(dem)
    .divide(ee.Image.constant(dem_range))
    .rename("lowland")
)

# AlphaEarth similarity to official Seoul flood positives
# AlphaEarth/Satellite Embedding은 위성영상 패턴을 압축한 다중 밴드 표현이다.
# 공식 침수 기준점 주변의 평균 임베딩과 각 픽셀의 임베딩 거리를 계산해,
# 침수 기준점과 위성영상 패턴이 비슷한 곳일수록 alpha_score가 높아지도록 만든다.
emb_collection = (
    ee.ImageCollection("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL")
    .filterDate(f"{YEAR}-01-01", f"{YEAR+1}-01-01")
    .filterBounds(seoul)
)
emb_count = emb_collection.size().getInfo()
print("AlphaEarth intersecting tiles:", emb_count)
if emb_count == 0:
    raise ValueError(f"AlphaEarth annual embeddings not found for Seoul in {YEAR}.")

emb = emb_collection.mosaic().clip(seoul)
band_names = emb.bandNames().getInfo()
print("AlphaEarth bands:", band_names[:10], "...")

# Hybrid-basic을 기준 모델로 고정한다.
valid_band_names = band_names
emb_valid = emb.select(valid_band_names).unmask(0)
static_feature_image = ee.Image.cat([slope, hnd, log_upa, water_occ, built, lowland])
static_feature_bands = static_feature_image.bandNames().getInfo()
alpha_feature_bands = ["alpha_score"]
hybrid_feature_bands = static_feature_bands + alpha_feature_bands
print("Fixed hybrid model bands:", hybrid_feature_bands)

def make_negative_mask(negative_buffer_m):
    """침수점 주변과 기존 수역을 제외한 음성 후보 영역을 만든다."""
    positive_exclusion_mask = ee.Image.constant(0).byte().paint(
        positive_geom.buffer(negative_buffer_m),
        1,
    )
    return (
        ee.Image.constant(1)
        .clip(seoul)
        .updateMask(water_occ.lt(0.2))
        .updateMask(dw_water.lt(0.25))
        .updateMask(positive_exclusion_mask.eq(0))
        .rename("negative_mask")
    )


def make_alpha_score(reference_fc):
    """해당 fold의 학습 양성점만 사용해 AlphaEarth 유사도 feature를 만든다."""
    reference_geom = reference_fc.geometry()
    reference_mean = ee.Dictionary(
        emb.clip(reference_geom)
        .unmask(0)
        .reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=reference_geom,
            scale=ANALYSIS_SCALE,
            maxPixels=1e9,
            bestEffort=True,
            tileScale=4,
        )
    )
    mean_img = reference_mean.toImage(valid_band_names).rename(valid_band_names)
    alpha_distance = (
        emb_valid.subtract(mean_img)
        .pow(2)
        .reduce(ee.Reducer.sum())
        .sqrt()
        .rename("alpha_distance")
    )
    alpha_stats = alpha_distance.reduceRegion(
        reducer=ee.Reducer.percentile([5, 95]),
        geometry=seoul,
        scale=ANALYSIS_SCALE,
        maxPixels=1e9,
        bestEffort=True,
        tileScale=4,
    ).getInfo()
    alpha_p5 = alpha_stats["alpha_distance_p5"]
    alpha_p95 = alpha_stats["alpha_distance_p95"]
    return (
        ee.Image.constant(alpha_p95)
        .subtract(alpha_distance)
        .divide(ee.Image.constant(max(alpha_p95 - alpha_p5, 1e-6)))
        .clamp(0, 1)
        .rename("alpha_score")
    )


fold_base_cache = {}


def build_fold_base(validation_fold):
    if validation_fold in fold_base_cache:
        return fold_base_cache[validation_fold]

    fold_train_positive = positive_points.filter(ee.Filter.neq("fold", validation_fold))
    fold_valid_positive = positive_points.filter(ee.Filter.eq("fold", validation_fold))
    fold_train_area = spatial_fold.neq(validation_fold)
    fold_valid_area = spatial_fold.eq(validation_fold)
    alpha_score = make_alpha_score(fold_train_positive)
    feature_image = static_feature_image.addBands(alpha_score)
    fold_base = {
        "feature_image": feature_image,
        "alpha_score": alpha_score,
        "train_positive": fold_train_positive,
        "valid_positive": fold_valid_positive,
        "train_area_mask": fold_train_area,
        "valid_area_mask": fold_valid_area,
    }
    fold_base_cache[validation_fold] = fold_base
    return fold_base


def build_hybrid_inputs(validation_fold, positive_buffer_m):
    fold_base = build_fold_base(validation_fold)
    positive_train_mask = ee.Image.constant(0).byte().paint(
        fold_base["train_positive"].map(lambda f: f.buffer(positive_buffer_m)),
        1,
    )
    positive_valid_mask = ee.Image.constant(0).byte().paint(
        fold_base["valid_positive"].map(lambda f: f.buffer(positive_buffer_m)),
        1,
    )
    return {
        **fold_base,
        "positive_train_mask": positive_train_mask,
        "positive_valid_mask": positive_valid_mask,
    }


def sample_split(feature_image, positive_mask, negative_mask, area_mask, seed):
    """공간 fold별 양성/음성 영역에서 학습 또는 검증 샘플을 만든다."""
    split_positive_mask = positive_mask.updateMask(area_mask)
    split_negative_mask = negative_mask.updateMask(area_mask)
    label_image = (
        ee.Image.constant(0)
        .clip(seoul)
        .where(split_positive_mask.unmask(0).eq(1), 1)
        .rename("label")
    )
    sampling_mask = (
        split_negative_mask.unmask(0).add(split_positive_mask.unmask(0)).gt(0)
    )
    split_image = feature_image.addBands(label_image).updateMask(sampling_mask)
    return split_image.stratifiedSample(
        numPoints=0,
        classBand="label",
        classValues=[0, 1],
        classPoints=[NEGATIVE_POINTS, POSITIVE_SAMPLE_POINTS],
        region=seoul,
        scale=ANALYSIS_SCALE,
        geometries=True,
        seed=seed,
        tileScale=4,
    )


def train_hybrid(train_fc):
    return train_rf(train_fc, hybrid_feature_bands)


def train_rf(train_fc, input_bands):
    return (
        ee.Classifier.smileRandomForest(
            numberOfTrees=100,
            variablesPerSplit=min(3, len(input_bands)),
            minLeafPopulation=2,
            bagFraction=0.7,
            seed=13,
        )
        .setOutputMode("PROBABILITY")
        .train(train_fc, "label", input_bands)
    )


def evaluate_fc(fc, classifier):
    evaluated = fc.classify(classifier, "probability").map(
        lambda f: f.set("predicted", ee.Number(f.get("probability")).gte(0.5).int())
    )
    return evaluated.errorMatrix("label", "predicted")


def classifier_importance(classifier, input_bands):
    """RF 변수 중요도와 정규화된 중요도를 반환한다."""
    explain_info = classifier.explain().getInfo()
    raw_importance = explain_info.get("importance", {})
    importance = {
        band: float(raw_importance.get(band, 0))
        for band in input_bands
    }
    total = sum(importance.values())
    normalized = {
        band: (value / total if total else 0)
        for band, value in importance.items()
    }
    return importance, normalized


def run_hybrid_fold(
    validation_fold,
    positive_buffer_m=POSITIVE_BUFFER_M,
    negative_buffer_m=NEGATIVE_BUFFER_M,
    include_map_outputs=False,
    include_importance=False,
    input_bands=None,
):
    if input_bands is None:
        input_bands = hybrid_feature_bands

    fold_inputs = build_hybrid_inputs(validation_fold, positive_buffer_m)
    negative_mask = make_negative_mask(negative_buffer_m)
    train_fc = sample_split(
        fold_inputs["feature_image"],
        fold_inputs["positive_train_mask"],
        negative_mask,
        fold_inputs["train_area_mask"],
        700 + validation_fold,
    )
    valid_fc = sample_split(
        fold_inputs["feature_image"],
        fold_inputs["positive_valid_mask"],
        negative_mask,
        fold_inputs["valid_area_mask"],
        1700 + validation_fold,
    )
    train_count = train_fc.size().getInfo()
    valid_count = valid_fc.size().getInfo()
    if train_count == 0 or valid_count == 0:
        raise ValueError(f"Fold {validation_fold}에서 학습/검증 샘플을 만들지 못했습니다.")

    classifier = train_rf(train_fc, input_bands)
    train_conf = evaluate_fc(train_fc, classifier)
    valid_conf = evaluate_fc(valid_fc, classifier)
    result = {
        "fold": validation_fold,
        "positive_buffer_m": positive_buffer_m,
        "negative_buffer_m": negative_buffer_m,
        "train_positive_count": fold_inputs["train_positive"].size().getInfo(),
        "valid_positive_count": fold_inputs["valid_positive"].size().getInfo(),
        "train_sample_count": train_count,
        "valid_sample_count": valid_count,
        "train_label_histogram": train_fc.aggregate_histogram("label").getInfo(),
        "valid_label_histogram": valid_fc.aggregate_histogram("label").getInfo(),
        "train_confusion": train_conf.getInfo(),
        "valid_confusion": valid_conf.getInfo(),
        "valid_accuracy": valid_conf.accuracy().getInfo(),
        "valid_kappa": valid_conf.kappa().getInfo(),
        "classifier": classifier,
        **fold_inputs,
    }

    if include_importance:
        importance, normalized_importance = classifier_importance(classifier, input_bands)
        result.update(
            {
                "importance": importance,
                "importance_normalized": normalized_importance,
            }
        )

    if include_map_outputs:
        probability = (
            fold_inputs["feature_image"]
            .select(input_bands)
            .classify(classifier, "flood_prob")
            .rename("flood_prob")
            .clip(seoul)
        )
        threshold = ee.Number(
            probability.reduceRegion(
                reducer=ee.Reducer.percentile([HOTSPOT_PERCENTILE]),
                geometry=seoul,
                scale=ANALYSIS_SCALE,
                maxPixels=1e9,
            ).get("flood_prob")
        )
        hotspots = probability.gte(threshold).selfMask().rename("hotspots")
        prob_stats = probability.reduceRegion(
            reducer=ee.Reducer.minMax().combine(
                reducer2=ee.Reducer.mean(),
                sharedInputs=True,
            ),
            geometry=seoul,
            scale=ANALYSIS_SCALE,
            maxPixels=1e9,
        ).getInfo()
        hotspot_area = (
            ee.Image.pixelArea()
            .divide(1e6)
            .updateMask(hotspots)
            .reduceRegion(
                reducer=ee.Reducer.sum(),
                geometry=seoul,
                scale=ANALYSIS_SCALE,
                maxPixels=1e9,
            )
            .getInfo()
        )
        result.update(
            {
                "probability": probability,
                "threshold": threshold.getInfo(),
                "hotspots": hotspots,
                "prob_stats": prob_stats,
                "hotspot_area": hotspot_area,
            }
        )

    return result


def mean(values):
    return sum(values) / max(len(values), 1)


def sample_std(values):
    if len(values) < 2:
        return 0
    avg = mean(values)
    return (sum((value - avg) ** 2 for value in values) / (len(values) - 1)) ** 0.5


def summarize_validation(rows):
    accuracies = [row["valid_accuracy"] for row in rows]
    kappas = [row["valid_kappa"] for row in rows]
    return {
        "accuracy_mean": mean(accuracies),
        "accuracy_std": sample_std(accuracies),
        "kappa_mean": mean(kappas),
        "kappa_std": sample_std(kappas),
    }


def summarize_importance(rows, input_bands):
    """fold별 정규화 변수 중요도의 평균과 표준편차를 요약한다."""
    summary = []
    rows_with_importance = [
        row for row in rows if "importance_normalized" in row
    ]
    for band in input_bands:
        values = [
            row["importance_normalized"].get(band, 0)
            for row in rows_with_importance
        ]
        summary.append(
            {
                "feature": band,
                "importance_mean": mean(values),
                "importance_std": sample_std(values),
            }
        )
    return sorted(summary, key=lambda row: row["importance_mean"], reverse=True)


def buffer_sensitivity_configs():
    configs = []
    seen = set()

    def add_config(kind, positive_buffer_m, negative_buffer_m):
        key = (positive_buffer_m, negative_buffer_m)
        if key in seen:
            return
        seen.add(key)
        configs.append(
            {
                "kind": kind,
                "positive_buffer_m": positive_buffer_m,
                "negative_buffer_m": negative_buffer_m,
            }
        )

    for positive_buffer_m in POSITIVE_BUFFER_SWEEP_M:
        add_config("positive_buffer", positive_buffer_m, NEGATIVE_BUFFER_M)
    for negative_buffer_m in NEGATIVE_BUFFER_SWEEP_M:
        add_config("negative_buffer", POSITIVE_BUFFER_M, negative_buffer_m)
    return configs


def run_buffer_sensitivity():
    sensitivity_folds = [
        fold for fold in BUFFER_SENSITIVITY_FOLDS if 0 <= fold < SPATIAL_FOLDS
    ]
    if not sensitivity_folds:
        print("Buffer sensitivity skipped: no valid folds configured.")
        return []

    print("\nHybrid-basic buffer sensitivity:")
    print(
        "Sensitivity folds/config:",
        {
            "folds": sensitivity_folds,
            "positive_sweep_m": POSITIVE_BUFFER_SWEEP_M,
            "negative_sweep_m": NEGATIVE_BUFFER_SWEEP_M,
        },
    )

    sensitivity_rows = []
    for config in buffer_sensitivity_configs():
        fold_rows = [
            run_hybrid_fold(
                fold,
                positive_buffer_m=config["positive_buffer_m"],
                negative_buffer_m=config["negative_buffer_m"],
                include_map_outputs=False,
            )
            for fold in sensitivity_folds
        ]
        summary = summarize_validation(fold_rows)
        row = {
            **config,
            "folds": sensitivity_folds,
            **summary,
        }
        sensitivity_rows.append(row)
        print(row)
    return sensitivity_rows


def run_spatial_cv(model_name, input_bands, include_importance=False):
    print(f"\n{model_name} spatial cross-validation:")
    rows = []
    for fold in range(SPATIAL_FOLDS):
        result = run_hybrid_fold(
            fold,
            include_map_outputs=False,
            include_importance=include_importance,
            input_bands=input_bands,
        )
        rows.append(result)
        print(
            {
                "fold": result["fold"],
                "train_pos": result["train_positive_count"],
                "valid_pos": result["valid_positive_count"],
                "valid_accuracy": result["valid_accuracy"],
                "valid_kappa": result["valid_kappa"],
                "valid_confusion": result["valid_confusion"],
            }
        )
    return rows


cv_results = run_spatial_cv(
    "Hybrid-basic",
    hybrid_feature_bands,
    include_importance=True,
)

cv_summary = summarize_validation(cv_results)
print("Hybrid-basic CV summary:", cv_summary)
importance_summary = summarize_importance(cv_results, hybrid_feature_bands)
print("Hybrid-basic normalized feature importance summary:")
for row in importance_summary:
    print(row)

no_water_feature_bands = [
    band for band in hybrid_feature_bands if band != "water_occ"
]
no_water_cv_results = run_spatial_cv(
    "Hybrid-no-water-occ",
    no_water_feature_bands,
    include_importance=False,
)
no_water_cv_summary = summarize_validation(no_water_cv_results)
print("Hybrid-no-water-occ CV summary:", no_water_cv_summary)
print(
    "Water occurrence ablation:",
    {
        "baseline_kappa_mean": cv_summary["kappa_mean"],
        "no_water_kappa_mean": no_water_cv_summary["kappa_mean"],
        "delta_kappa": no_water_cv_summary["kappa_mean"] - cv_summary["kappa_mean"],
        "baseline_accuracy_mean": cv_summary["accuracy_mean"],
        "no_water_accuracy_mean": no_water_cv_summary["accuracy_mean"],
        "delta_accuracy": no_water_cv_summary["accuracy_mean"] - cv_summary["accuracy_mean"],
    },
)

selected_model_name = "Hybrid-basic"
selected_feature_bands = hybrid_feature_bands
if no_water_cv_summary["kappa_mean"] >= cv_summary["kappa_mean"]:
    selected_model_name = "Hybrid-no-water-occ"
    selected_feature_bands = no_water_feature_bands
    print("Selected feature set: water_occ removed from RF inputs.")
else:
    print("Selected feature set: keeping water_occ in RF inputs.")

if RUN_BUFFER_SENSITIVITY:
    buffer_sensitivity_rows = run_buffer_sensitivity()
else:
    buffer_sensitivity_rows = []

hybrid_result = run_hybrid_fold(
    VALIDATION_FOLD,
    include_map_outputs=True,
    input_bands=selected_feature_bands,
)
seoul_probability = hybrid_result["probability"]
alpha_score = hybrid_result["alpha_score"]
hotspots = hybrid_result["hotspots"]
threshold = hybrid_result["threshold"]
prob_stats = hybrid_result["prob_stats"]
hotspot_area = hybrid_result["hotspot_area"]
valid_area_mask = hybrid_result["valid_area_mask"]
valid_positive_points = hybrid_result["valid_positive"]

print(f"\nSelected map fold: {VALIDATION_FOLD}")
print("Selected map model:", selected_model_name)
print("Selected map bands:", selected_feature_bands)
print("Selected fold validation accuracy:", hybrid_result["valid_accuracy"])
print("Selected fold validation kappa:", hybrid_result["valid_kappa"])
print("Seoul probability stats:", prob_stats)
print(f"P{HOTSPOT_PERCENTILE} threshold:", threshold)
print("Hotspot area (km²):", hotspot_area)

# 결과 확인용 대화형 HTML 지도를 만든다.
# 경계, 기준점, AlphaEarth 임베딩, Hybrid 확률, hotspot 레이어를 함께 저장한다.
centroid = seoul.centroid().coordinates().getInfo()
lon, lat = centroid[0], centroid[1]
rgb_bands = valid_band_names[:3]

Map = geemap.Map(center=[lat, lon], zoom=11)
Map.addLayer(
    ee.Image().paint(seoul_fc, 1, 2),
    {"palette": ["cyan"]},
    "Seoul boundary",
)
Map.addLayer(
    positive_points,
    {"color": "#542788"},
    "Official Seoul flood positives",
)
Map.addLayer(
    valid_area_mask.selfMask(),
    {"palette": ["#ffd92f"], "opacity": 0.25},
    f"Validation spatial fold {VALIDATION_FOLD}",
)
Map.addLayer(
    valid_positive_points,
    {"color": "#1b9e77"},
    "Held-out validation positives",
)
Map.addLayer(
    emb.select(rgb_bands),
    {"min": -0.3, "max": 0.3},
    f"AlphaEarth {YEAR} ({','.join(rgb_bands)})",
)
Map.addLayer(
    alpha_score,
    {"min": 0, "max": 1, "palette": ["#f7f7f7", "#fdb863", "#e66101", "#b2182b"]},
    "AlphaEarth flood similarity",
)
Map.addLayer(
    seoul_probability,
    {"min": 0, "max": 1, "palette": ["#f7fbff", "#6baed6", "#2171b5", "#08306b"]},
    "Hybrid RF flood probability",
)
Map.addLayer(
    hotspots,
    {"palette": ["red"]},
    f"Hybrid hotspots (top {100 - HOTSPOT_PERCENTILE}%)",
)
Map.addLayerControl()
Map.to_html(OUTPUT_HTML)
print(f"Saved: {OUTPUT_HTML}")
