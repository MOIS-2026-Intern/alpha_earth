import json
import os

import ee
import geemap


def load_env_file(path=".env"):
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

PROJECT_ID = os.environ.get("EE_PROJECT_ID")
if not PROJECT_ID:
    raise ValueError("EE_PROJECT_ID is missing. Set it in .env or your shell environment.")

YEAR = int(os.environ.get("YEAR", "2024"))
OUTPUT_HTML = os.environ.get("OUTPUT_HTML", "seoul_flood_risk_rf.html")
ANALYSIS_SCALE = int(os.environ.get("ANALYSIS_SCALE", "30"))
SEOUL_REFERENCE_GEOJSON = os.environ.get(
    "SEOUL_REFERENCE_GEOJSON", "seoul_flood_reference_points.geojson"
)
POSITIVE_POINTS = int(os.environ.get("POSITIVE_POINTS", "200"))
NEGATIVE_POINTS = int(os.environ.get("NEGATIVE_POINTS", "200"))
NEGATIVE_BUFFER_M = int(os.environ.get("NEGATIVE_BUFFER_M", "300"))
POSITIVE_BUFFER_M = int(os.environ.get("POSITIVE_BUFFER_M", "60"))
HOTSPOT_PERCENTILE = int(os.environ.get("HOTSPOT_PERCENTILE", "95"))

ee.Initialize(project=PROJECT_ID)


def add_random_column(fc, seed):
    return fc.randomColumn(columnName="rand", seed=seed)


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

with open(SEOUL_REFERENCE_GEOJSON, "r", encoding="utf-8") as f:
    seoul_reference_geojson = json.load(f)

positive_features = [
    ee.Feature(
        ee.Geometry.Point(feature["geometry"]["coordinates"]),
        {
            **feature["properties"],
            "label": 1,
        },
    )
    for feature in seoul_reference_geojson["features"][:POSITIVE_POINTS]
]
positive_points = ee.FeatureCollection(positive_features)
positive_geom = positive_points.geometry()
print("Positive flood points:", len(positive_features))

# -------------------------------------------------
# Feature stack for Seoul
# -------------------------------------------------
dem = ee.Image("USGS/SRTMGL1_003").clip(seoul)
slope = ee.Terrain.slope(dem).rename("slope")
merit = ee.Image("MERIT/Hydro/v1_0_1").clip(seoul)
hnd = merit.select("hnd").rename("hnd")
log_upa = merit.select("upa").add(1).log().rename("log_upa")

water_occ = (
    ee.Image("JRC/GSW1_4/GlobalSurfaceWater")
    .select("occurrence")
    .clip(seoul)
    .unmask(0)
    .divide(100)
    .rename("water_occ")
)

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

emb_reference = emb.clip(positive_geom).unmask(0)
reference_mean = ee.Dictionary(
    emb_reference.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=positive_geom,
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
alpha_score = (
    ee.Image.constant(alpha_p95)
    .subtract(alpha_distance)
    .divide(ee.Image.constant(max(alpha_p95 - alpha_p5, 1e-6)))
    .clamp(0, 1)
    .rename("alpha_score")
)
print("AlphaEarth similarity feature: enabled")

feature_image = ee.Image.cat(
    [slope, hnd, log_upa, water_occ, built, lowland, alpha_score]
)
feature_bands = feature_image.bandNames().getInfo()
print("Feature bands:", feature_bands)

# -------------------------------------------------
# Negative background points inside Seoul
# -------------------------------------------------
positive_buffer_mask = ee.Image.constant(0).byte().paint(
    positive_geom.buffer(NEGATIVE_BUFFER_M),
    1,
)
positive_train_mask = ee.Image.constant(0).byte().paint(
    positive_points.map(lambda f: f.buffer(POSITIVE_BUFFER_M)),
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
candidate_negative_points = ee.FeatureCollection.randomPoints(
    region=seoul,
    points=NEGATIVE_POINTS * 20,
    seed=42,
    maxError=100,
)
negative_points = (
    negative_mask.sampleRegions(
        collection=candidate_negative_points,
        scale=ANALYSIS_SCALE,
        geometries=True,
        tileScale=4,
    )
    .filter(ee.Filter.notNull(["negative_mask"]))
    .limit(NEGATIVE_POINTS)
    .map(lambda f: ee.Feature(f.geometry(), {"label": 0}))
)
negative_count = negative_points.size().getInfo()
print("Negative background points:", negative_count)
if negative_count == 0:
    raise ValueError("Negative background points could not be sampled in Seoul.")

# -------------------------------------------------
# Train RF on Seoul positives/background
# -------------------------------------------------
label_image = ee.Image.constant(0).clip(seoul).where(positive_train_mask.eq(1), 1).rename(
    "label"
)
sampling_mask = negative_mask.unmask(0).add(positive_train_mask).gt(0)
training_image = feature_image.addBands(label_image).updateMask(sampling_mask)

sampled = training_image.stratifiedSample(
    numPoints=0,
    classBand="label",
    classValues=[0, 1],
    classPoints=[NEGATIVE_POINTS, POSITIVE_POINTS],
    region=seoul,
    scale=ANALYSIS_SCALE,
    geometries=False,
    seed=7,
    tileScale=4,
)

sample_count = sampled.size().getInfo()
print("Sampled points with features:", sample_count)

sampled = add_random_column(sampled, 7)
train_fc = sampled.filter(ee.Filter.lt("rand", 0.8))
valid_fc = sampled.filter(ee.Filter.gte("rand", 0.8))
train_count = train_fc.size().getInfo()
valid_count = valid_fc.size().getInfo()
print("Train / valid:", train_count, valid_count)

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
seoul_probability = (
    feature_image.classify(classifier, "flood_prob")
    .rename("flood_prob")
    .clip(seoul)
)

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
