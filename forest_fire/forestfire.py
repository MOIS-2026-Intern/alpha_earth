import json
import math
import os
import time

import ee
import geemap
import requests
import urllib3


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
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SOURCE_DIR = os.path.join(SCRIPT_DIR, "source")
HISTORY_CACHE_PATH = os.environ.get(
    "FORESTFIRE_HISTORY_CACHE",
    os.path.join(SOURCE_DIR, "forestfire_history.json"),
)
OUTPUT_HTML = os.environ.get(
    "OUTPUT_HTML",
    os.path.join(SCRIPT_DIR, "forestfire_similarity.html"),
)
ANALYSIS_SCALE = int(os.environ.get("ANALYSIS_SCALE", "30"))

# -------------------------------------------------
# Region selection
#   REGION 변수만 바꾸면 다른 시도로 전환됩니다. (예: "서울", "충청남도")
#   환경변수 REGION 으로도 덮어쓸 수 있습니다.
# -------------------------------------------------
REGION_PRESETS = {
    "충청남도": ["Chungcheongnam", "South Chungcheong", "충청남도"],
    "서울": ["Seoul", "서울"],
    "경기도": ["Gyeonggi", "경기도"],
    "강원도": ["Gangwon", "강원"],
    "충청북도": ["Chungcheongbuk", "North Chungcheong", "충청북도"],
    "전라남도": ["Jeollanam", "South Jeolla", "전라남도"],
    "전라북도": ["Jeollabuk", "North Jeolla", "전라북도"],
    "경상남도": ["Gyeongsangnam", "South Gyeongsang", "경상남도"],
    "경상북도": ["Gyeongsangbuk", "North Gyeongsang", "경상북도"],
    "제주도": ["Jeju", "제주"],
    "부산": ["Busan", "부산"],
    "대구": ["Daegu", "대구"],
    "인천": ["Incheon", "인천"],
    "광주": ["Gwangju", "광주"],
    "대전": ["Daejeon", "대전"],
    "울산": ["Ulsan", "울산"],
    "세종": ["Sejong", "세종"]
}
REGION = os.environ.get("REGION", "경상남도")
REGION_NAME_PATTERNS = REGION_PRESETS.get(REGION, [REGION])

FORESTFIRE_API_URL = (
    "https://www.safetydata.go.kr/V2/api/DSSP-IF-10854"
)
SERVICE_KEY = os.environ.get("FORESTFIRE_SERVICE_KEY")
if not SERVICE_KEY:
    raise ValueError("FORESTFIRE_SERVICE_KEY is missing. Set it in .env or your shell environment.")
PAGE_SIZE = int(os.environ.get("FORESTFIRE_PAGE_SIZE", "1000"))

ee.Initialize(project=PROJECT_ID)


# -------------------------------------------------
# Coordinate helpers (EPSG:3857 -> EPSG:4326)
# -------------------------------------------------
_R = 6378137.0


def mercator_to_lonlat(x, y):
    lon = x / _R * (180.0 / math.pi)
    lat = (math.atan(math.exp(y / _R)) * 2.0 - math.pi / 2.0) * (180.0 / math.pi)
    return lon, lat


# -------------------------------------------------
# Fetch every page of forest fire history (cached)
# -------------------------------------------------
def fetch_forestfire_history():
    if os.path.exists(HISTORY_CACHE_PATH):
        with open(HISTORY_CACHE_PATH, "r", encoding="utf-8") as f:
            cached = json.load(f)
        print(f"Loaded cached forest fire history: {len(cached)} records")
        return cached

    session = requests.Session()
    all_records = []
    page = 1
    total_count = None
    while True:
        params = {
            "serviceKey": SERVICE_KEY,
            "pageNo": page,
            "numOfRows": PAGE_SIZE,
            "returnType": "json",
        }
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        resp = session.get(FORESTFIRE_API_URL, params=params, timeout=60)
        resp.raise_for_status()
        payload = resp.json()
        header = payload.get("header", {})
        if header.get("resultCode") not in (None, "00"):
            raise RuntimeError(f"API error: {header}")
        body = payload.get("body") or []
        if total_count is None:
            total_count = int(payload.get("totalCount", 0))
            print(f"Forest fire history totalCount: {total_count}")
        all_records.extend(body)
        print(f"  page {page}: {len(body)} records (cumulative {len(all_records)})")
        if not body or len(all_records) >= total_count:
            break
        page += 1
        time.sleep(0.2)

    with open(HISTORY_CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(all_records, f, ensure_ascii=False)
    print(f"Saved cache: {HISTORY_CACHE_PATH}")
    return all_records


def history_records_to_lonlat(records):
    points = []
    for rec in records:
        x = rec.get("XCRD")
        y = rec.get("YCRD")
        if x is None or y is None:
            continue
        try:
            lon, lat = mercator_to_lonlat(float(x), float(y))
        except (TypeError, ValueError):
            continue
        points.append((lon, lat, rec))
    return points


# -------------------------------------------------
# Build reference points within the analysis region
# -------------------------------------------------
def reference_points_within(region_geom, lonlat_records):
    bounds = region_geom.bounds().getInfo()["coordinates"][0]
    xs = [pt[0] for pt in bounds]
    ys = [pt[1] for pt in bounds]
    minx, maxx = min(xs), max(xs)
    miny, maxy = min(ys), max(ys)

    candidate_features = []
    for lon, lat, rec in lonlat_records:
        if minx <= lon <= maxx and miny <= lat <= maxy:
            candidate_features.append(
                ee.Feature(
                    ee.Geometry.Point([lon, lat]),
                    {
                        "OCRN_YMD": rec.get("OCRN_YMD"),
                        "RSN": rec.get("RSN"),
                        "ADDR": rec.get("ADDR"),
                        "ID": rec.get("ID"),
                    },
                )
            )
    print(f"Reference points within region bounding box: {len(candidate_features)}")
    if not candidate_features:
        raise ValueError("No forest fire history points fall within the region.")
    candidates_fc = ee.FeatureCollection(candidate_features)
    inside_fc = candidates_fc.filterBounds(region_geom)
    inside_count = inside_fc.size().getInfo()
    print(f"Reference points strictly inside region polygon: {inside_count}")
    if inside_count == 0:
        print("Falling back to bounding-box candidates for reference vector.")
        return candidates_fc
    return inside_fc


# -------------------------------------------------
# Pipeline
# -------------------------------------------------
records = fetch_forestfire_history()
lonlat_records = history_records_to_lonlat(records)
print(f"Records with valid coordinates: {len(lonlat_records)}")

adm1 = ee.FeatureCollection("WM/geoLab/geoBoundaries/600/ADM1")
kor_adm1 = adm1.filter(ee.Filter.eq("shapeGroup", "KOR"))
region_fc = kor_adm1.filter(
    ee.Filter.Or(*[
        ee.Filter.stringContains("shapeName", pattern)
        for pattern in REGION_NAME_PATTERNS
    ])
)
region_count = region_fc.size().getInfo()
print(f"Matched region features for '{REGION}': {region_count}")
if region_count == 0:
    raise ValueError(f"'{REGION}' 경계를 찾지 못했습니다. REGION_PRESETS 또는 REGION 환경변수를 확인하세요.")
region_geom = region_fc.geometry()

reference_fc = reference_points_within(region_geom, lonlat_records)
reference_geom = reference_fc.geometry().buffer(ANALYSIS_SCALE)

emb_collection = (
    ee.ImageCollection("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL")
    .filterDate(f"{YEAR}-01-01", f"{YEAR + 1}-01-01")
    .filterBounds(region_geom)
)
emb_count = emb_collection.size().getInfo()
print(f"AlphaEarth intersecting tiles: {emb_count}")
if emb_count == 0:
    raise ValueError(f"AlphaEarth annual embeddings not found for region in {YEAR}.")

emb = emb_collection.mosaic().clip(region_geom)
band_names = emb.bandNames().getInfo()
print(f"AlphaEarth bands: {band_names[:10]} ... ({len(band_names)} total)")

emb_reference = emb.clip(reference_geom).unmask(0)
reference_mean = ee.Dictionary(
    emb_reference.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=reference_geom,
        scale=ANALYSIS_SCALE,
        maxPixels=1e9,
        bestEffort=True,
        tileScale=4,
    )
)
reference_mean_info = reference_mean.getInfo()
valid_band_names = [b for b in band_names if reference_mean_info.get(b) is not None]
print(f"Valid AlphaEarth bands: {len(valid_band_names)}")
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
    .clip(region_geom)
)

alpha_stats = alpha_distance.reduceRegion(
    reducer=ee.Reducer.percentile([5, 95]),
    geometry=region_geom,
    scale=ANALYSIS_SCALE,
    maxPixels=1e9,
    bestEffort=True,
    tileScale=4,
).getInfo()
alpha_p5 = alpha_stats["alpha_distance_p5"]
alpha_p95 = alpha_stats["alpha_distance_p95"]
print(f"alpha_distance p5/p95: {alpha_p5:.4f} / {alpha_p95:.4f}")

alpha_score = (
    ee.Image.constant(alpha_p95)
    .subtract(alpha_distance)
    .divide(ee.Image.constant(max(alpha_p95 - alpha_p5, 1e-6)))
    .clamp(0, 1)
    .rename("alpha_score")
    .clip(region_geom)
)

centroid = region_geom.centroid(maxError=1).coordinates().getInfo()
lon, lat = centroid[0], centroid[1]


# 5단계로 구간 나누기
alpha_score_binned = (
    alpha_score.multiply(5).floor().min(4).divide(4).rename("alpha_score_binned")
)
alpha_score_display = alpha_score_binned.updateMask(alpha_score_binned)

fire_palette_5 = [
    "#FEE5D9",
    "#FCAE91",
    "#FB6A4A",
    "#DE2D26",
    "#CB181D",
]

Map = geemap.Map(center=[lat, lon], zoom=10)
Map.addLayer(
    alpha_score_display,
    {
        "min": 0,
        "max": 1,
        "palette": fire_palette_5,
    },
    "AlphaEarth forest fire similarity",
)
Map.addLayer(
    ee.Image().paint(region_fc, 1, 2),
    {"palette": ["#000000"]},
    "Region boundary",
)
Map.addLayer(
    reference_fc,
    {"color": "#542788"},
    "Past forest fire points (reference)",
)
Map.addLayerControl()
Map.to_html(OUTPUT_HTML)
print(f"Saved: {OUTPUT_HTML}")
