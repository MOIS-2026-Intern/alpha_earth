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

# 학습 양성점 영역에서만 AlphaEarth 밴드별 평균값을 구한다.
# 검증 fold의 양성점은 prototype 계산에서 제외해 AlphaEarth feature 누수를 막는다.
emb_reference = emb.clip(train_positive_geom).unmask(0)
reference_mean = ee.Dictionary(
    emb_reference.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=train_positive_geom,
        scale=ANALYSIS_SCALE,
        maxPixels=1e9,
        bestEffort=True,
        tileScale=4,
    )
)
reference_mean_info = reference_mean.getInfo()
valid_band_names = [b for b in band_names if b in reference_mean_info]
print("Valid AlphaEarth bands:", len(valid_band_names))
if not valid_band_names:
    raise ValueError("AlphaEarth reference mean is empty.")

# 각 픽셀과 양성 기준점 평균 임베딩 사이의 유클리드 거리를 계산한다.
emb_valid = emb.select(valid_band_names).unmask(0)
emb_mean_img = reference_mean.toImage(valid_band_names).rename(valid_band_names)
alpha_distance = (
    emb_valid.subtract(emb_mean_img)
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
# 거리는 작을수록 기준점과 비슷하므로, 5~95 분위 범위로 뒤집어 0~1 유사도 점수로 변환한다.
alpha_score = (
    ee.Image.constant(alpha_p95)
    .subtract(alpha_distance)
    .divide(ee.Image.constant(max(alpha_p95 - alpha_p5, 1e-6)))
    .clamp(0, 1)
    .rename("alpha_score")
)
print("AlphaEarth similarity feature: enabled")

# 랜덤포레스트가 사용할 모든 설명변수를 하나의 다중 밴드 영상으로 묶는다.
feature_image = ee.Image.cat(
    [slope, hnd, log_upa, water_occ, built, lowland, alpha_score]
)
feature_bands = feature_image.bandNames().getInfo()
print("Feature bands:", feature_bands)

# -------------------------------------------------
# Train/validation samples inside Seoul
# -------------------------------------------------
# 음성(label=0) 후보 영역은 기존 수역과 전체 공식 침수 기준점 주변을 제외한 배경 지역으로 둔다.
# 이후 train/validation 공간 fold를 적용해 서로 다른 공간 블록에서 샘플을 뽑는다.
positive_buffer_mask = ee.Image.constant(0).byte().paint(
    positive_geom.buffer(NEGATIVE_BUFFER_M),
    1,
)
positive_train_mask = ee.Image.constant(0).byte().paint(
    train_positive_points.map(lambda f: f.buffer(POSITIVE_BUFFER_M)),
    1,
)
positive_valid_mask = ee.Image.constant(0).byte().paint(
    valid_positive_points.map(lambda f: f.buffer(POSITIVE_BUFFER_M)),
    1,
)
negative_mask = (
    ee.Image.constant(1)
    .clip(seoul)
    .updateMask(water_occ.lt(0.2))
    .updateMask(dw_water.lt(0.25))
    .updateMask(positive_buffer_mask.eq(0))
    .rename("negative_mask")
)


def sample_split(positive_mask, area_mask, seed):
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


# 양성/음성에서 지정한 개수만큼 계층 샘플링해 학습/검증 테이블을 별도로 만든다.
train_fc = sample_split(positive_train_mask, train_area_mask, 7)
valid_fc = sample_split(positive_valid_mask, valid_area_mask, 17)
train_count = train_fc.size().getInfo()
valid_count = valid_fc.size().getInfo()
print("Train / validation sampled pixels:", train_count, valid_count)
print("Train label histogram:", train_fc.aggregate_histogram("label").getInfo())
print("Validation label histogram:", valid_fc.aggregate_histogram("label").getInfo())
if train_count == 0 or valid_count == 0:
    raise ValueError("공간 fold 기반 학습/검증 샘플을 만들지 못했습니다.")

# 랜덤포레스트 분류기를 학습한다.
# 출력 모드를 PROBABILITY로 설정해 각 픽셀의 label=1 가능성을 0~1 확률처럼 받는다.
classifier = (
    ee.Classifier.smileRandomForest(
        numberOfTrees=100,
        variablesPerSplit=3,
        minLeafPopulation=2,
        bagFraction=0.7,
        seed=13,
    )
    .setOutputMode("PROBABILITY")
    .train(train_fc, "label", feature_bands)
)

# 학습/검증 데이터에서 0.5 기준으로 예측값을 만들고 혼동행렬, 정확도, kappa를 확인한다.
train_eval = train_fc.classify(classifier, "probability").map(
    lambda f: f.set("predicted", ee.Number(f.get("probability")).gte(0.5).int())
)
valid_eval = valid_fc.classify(classifier, "probability").map(
    lambda f: f.set("predicted", ee.Number(f.get("probability")).gte(0.5).int())
)
train_conf = train_eval.errorMatrix("label", "predicted")
valid_conf = valid_eval.errorMatrix("label", "predicted")
print("Train confusion matrix:", train_conf.getInfo())
print("Valid confusion matrix:", valid_conf.getInfo())
print("Valid accuracy:", valid_conf.accuracy().getInfo())
print("Valid kappa:", valid_conf.kappa().getInfo())

# -------------------------------------------------
# Predict flood susceptibility in Seoul
# -------------------------------------------------
# 학습된 모델을 서울 전체 픽셀에 적용해 침수 가능성 확률 지도를 만든다.
seoul_probability = (
    feature_image.classify(classifier, "flood_prob")
    .rename("flood_prob")
    .clip(seoul)
)

# 확률값 상위 HOTSPOT_PERCENTILE 분위 이상만 hotspot으로 표시한다.
# 기본값 95는 서울 전체 픽셀 중 상위 5%를 예상 침수 취약 지역으로 보는 설정이다.
threshold = ee.Number(
    seoul_probability.reduceRegion(
        reducer=ee.Reducer.percentile([HOTSPOT_PERCENTILE]),
        geometry=seoul,
        scale=ANALYSIS_SCALE,
        maxPixels=1e9,
    ).get("flood_prob")
)
hotspots = seoul_probability.gte(threshold).selfMask().rename("hotspots")

prob_stats = seoul_probability.reduceRegion(
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
print("Seoul probability stats:", prob_stats)
print(f"P{HOTSPOT_PERCENTILE} threshold:", threshold.getInfo())
print("Hotspot area (km²):", hotspot_area)

# 결과 확인용 대화형 HTML 지도를 만든다.
# 경계, 기준점, AlphaEarth 임베딩, AlphaEarth 유사도, RF 확률, hotspot 레이어를 함께 저장한다.
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
    "RF flood probability",
)
Map.addLayer(
    hotspots,
    {"palette": ["red"]},
    f"Hotspots (top {100 - HOTSPOT_PERCENTILE}%)",
)
Map.addLayerControl()
Map.to_html(OUTPUT_HTML)
print(f"Saved: {OUTPUT_HTML}")
