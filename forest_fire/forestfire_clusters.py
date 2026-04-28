"""산불 발생지 임베딩을 k-means로 군집화해 전국 유사도 지도를 생성하는 스크립트."""
import json
import math
import os
import time

import ee
import geemap
import requests
import urllib3


# ==============================
# 환경 변수 로딩
# ==============================
def load_env_file(path=".env"):
    """.env 파일을 읽어 os.environ에 주입."""
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

# ==============================
# 전역 상수 / 파라미터
# ==============================
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
    "OUTPUT_HTML_CLUSTERS",
    os.path.join(SCRIPT_DIR, "forestfire_similarity_clusters.html"),
)
ANALYSIS_SCALE = int(os.environ.get("ANALYSIS_SCALE", "30"))    # 픽셀 분석 해상도 (m)
K_CLUSTERS = int(os.environ.get("K_CLUSTERS", "10"))             # 클러스터 개수
SAMPLE_LIMIT = int(os.environ.get("SAMPLE_LIMIT", "3000"))      # 학습용 산불 지점 최대 수
NORMALIZE_SCALE = int(os.environ.get("NORMALIZE_SCALE", "500")) # 임계값 분위수 계산 해상도 (m)
TOP_PERCENT = int(os.environ.get("TOP_PERCENT", "10"))           # 상위 X% 만 빨갛게 표시

# 산불 이력 API (안전데이터광장)
FORESTFIRE_API_URL = "https://www.safetydata.go.kr/V2/api/DSSP-IF-10854"
SERVICE_KEY = os.environ.get("FORESTFIRE_SERVICE_KEY")
if not SERVICE_KEY:
    raise ValueError("FORESTFIRE_SERVICE_KEY is missing. Set it in .env or your shell environment.")
PAGE_SIZE = int(os.environ.get("FORESTFIRE_PAGE_SIZE", "1000"))

# ADM1 영문 shapeName -> 한국어 행정구역명 (지도 레이어 라벨용)
REGION_NAME_MAP = {
    "Chungcheongnam": "충청남도",
    "Chungcheongbuk": "충청북도",
    "Gyeonggi": "경기도",
    "Gangwon": "강원도",
    "Jeollanam": "전라남도",
    "Jeollabuk": "전라북도",
    "Gyeongsangnam": "경상남도",
    "Gyeongsangbuk": "경상북도",
    "Jeju": "제주도",
    "Seoul": "서울",
    "Busan": "부산",
    "Daegu": "대구",
    "Incheon": "인천",
    "Gwangju": "광주",
    "Daejeon": "대전",
    "Ulsan": "울산",
    "Sejong": "세종",
}


# ==============================
# 행정구역명 한국어 변환
# ==============================
def to_korean_region_name(shape_name):
    """ADM1 영문 shapeName에서 매칭되는 한국어명을 반환 (없으면 원문)."""
    if not shape_name:
        return "Unknown"
    for english, korean in REGION_NAME_MAP.items():
        if english.lower() in shape_name.lower():
            return korean
    return shape_name


ee.Initialize(project=PROJECT_ID)


# ==============================
# 좌표 변환 (Web Mercator → 위경도)
# ==============================
_R = 6378137.0  # 지구 반경 (m)


def mercator_to_lonlat(x, y):
    """EPSG:3857 좌표를 EPSG:4326 (lon, lat)로 변환."""
    lon = x / _R * (180.0 / math.pi)
    lat = (math.atan(math.exp(y / _R)) * 2.0 - math.pi / 2.0) * (180.0 / math.pi)
    return lon, lat


# ==============================
# 산불 이력 데이터 수집
# ==============================
def fetch_forestfire_history():
    # 안전데이터공유플랫폼 산불 이력 전체 페이지를 받아오고 로컬에 캐시
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

    os.makedirs(os.path.dirname(HISTORY_CACHE_PATH), exist_ok=True)
    with open(HISTORY_CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(all_records, f, ensure_ascii=False)
    print(f"Saved cache: {HISTORY_CACHE_PATH}")
    return all_records


# ==============================
# API 레코드 → EE Feature 변환
# ==============================
def history_records_to_features(records):
    """산불 이력 레코드(XCRD/YCRD)를 EE Point Feature 리스트로 변환."""
    features = []
    for rec in records:
        x = rec.get("XCRD")
        y = rec.get("YCRD")
        if x is None or y is None:
            continue
        try:
            lon, lat = mercator_to_lonlat(float(x), float(y))
        except (TypeError, ValueError):
            continue
        features.append(ee.Feature(ee.Geometry.Point([lon, lat])))
    return features


# ==============================
# 파이프라인 1: 산불 지점 + 한국 경계 준비
# ==============================
records = fetch_forestfire_history()
fire_features = history_records_to_features(records)
print(f"Fire history with valid coordinates: {len(fire_features)}")

# 한국 광역행정구역(ADM1) 경계 로드
adm1 = ee.FeatureCollection("WM/geoLab/geoBoundaries/600/ADM1")
kor_adm1 = adm1.filter(ee.Filter.eq("shapeGroup", "KOR"))
korea_geom = kor_adm1.geometry()
print("Loaded Korea ADM1 boundaries.")

# 한국 영토 안의 산불 지점만 사용, 학습용으로 무작위 SAMPLE_LIMIT개 추출
fire_fc_raw = ee.FeatureCollection(fire_features)
fire_fc = fire_fc_raw.filterBounds(korea_geom)
fire_fc_count = fire_fc.size().getInfo()
print(f"Fire points within Korea: {fire_fc_count}")

training_fc = fire_fc.randomColumn("rnd", 42).sort("rnd").limit(SAMPLE_LIMIT)
print(f"Training sample target: up to {SAMPLE_LIMIT}")

# ==============================
# 파이프라인 2: AlphaEarth 임베딩 로드 및 샘플링
# ==============================
emb_collection = (
    ee.ImageCollection("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL")
    .filterDate(f"{YEAR}-01-01", f"{YEAR + 1}-01-01")
    .filterBounds(korea_geom)
)
emb_count = emb_collection.size().getInfo()
print(f"AlphaEarth intersecting tiles: {emb_count}")
if emb_count == 0:
    raise ValueError(f"AlphaEarth annual embeddings not found for Korea in {YEAR}.")

# 64-band 임베딩 모자이크를 한국 경계로 자르고 마스크 영역은 0으로 채움
emb = emb_collection.mosaic().clip(korea_geom)
band_names = emb.bandNames().getInfo()
print(f"AlphaEarth bands: {len(band_names)} total")

emb_valid = emb.select(band_names).unmask(0)

# 산불 지점에서 임베딩 벡터 샘플링 (클러스터링 학습 데이터)
fire_samples = emb_valid.sampleRegions(
    collection=training_fc,
    scale=ANALYSIS_SCALE,
    tileScale=4,
    geometries=False,
)
sample_count = fire_samples.size().getInfo()
print(f"Sampled embedding vectors at fire points: {sample_count}")
if sample_count == 0:
    raise ValueError("No embedding samples produced from fire points.")

# ==============================
# 파이프라인 3: k-means 클러스터링
# ==============================
print(f"Training k-means with K={K_CLUSTERS}...")
clusterer = ee.Clusterer.wekaKMeans(K_CLUSTERS).train(
    features=fire_samples,
    inputProperties=band_names,
)
# 학습된 클러스터러로 산불 샘플들에 cluster 라벨(0..K-1) 부여
clustered_samples = fire_samples.cluster(clusterer)


# ==============================
# 군집별 중심점(평균 벡터) 계산
# ==============================
def cluster_centroid(k):
    """k번 군집 샘플들의 밴드별 평균 벡터를 ee.Dictionary로 반환."""
    subset = clustered_samples.filter(ee.Filter.eq("cluster", k))
    result = subset.reduceColumns(
        reducer=ee.Reducer.mean().repeat(len(band_names)),
        selectors=band_names,
    )
    mean_list = ee.List(result.get("mean"))
    return ee.Dictionary.fromLists(band_names, mean_list)


# ==============================
# 파이프라인 4: 픽셀 ↔ 중심점 거리 계산
# ==============================
print("Computing per-cluster mean vectors and distance images...")
distance_images = []
cluster_sizes = []
for k in range(K_CLUSTERS):
    cluster_sizes.append(clustered_samples.filter(ee.Filter.eq("cluster", k)).size())
    centroid = cluster_centroid(k)
    centroid_img = centroid.toImage(band_names).rename(band_names)
    # 유클리드 거리 = sqrt(Σ(픽셀_b − 중심점_b)²)
    d = emb_valid.subtract(centroid_img).pow(2).reduce(ee.Reducer.sum()).sqrt()
    distance_images.append(d)

sizes_info = ee.List(cluster_sizes).getInfo()
print(f"Cluster sizes: {sizes_info}")

# K개 거리 중 픽셀별 최솟값 = "가장 가까운 군집과의 거리"
min_distance = (
    ee.ImageCollection.fromImages(distance_images)
    .min()
    .rename("min_distance")
    .clip(korea_geom)
)

# ==============================
# 파이프라인 5: 상위 X% 임계값 마스킹
# ==============================
print(f"Computing top-{TOP_PERCENT}% similarity threshold...")
threshold_stats = min_distance.reduceRegion(
    reducer=ee.Reducer.percentile([TOP_PERCENT]),
    geometry=korea_geom,
    scale=NORMALIZE_SCALE,
    maxPixels=1e10,
    bestEffort=True,
    tileScale=4,
).getInfo()
print(f"Threshold stats: {threshold_stats}")
# Reducer.percentile([N])의 출력 키 이름이 환경별로 다르므로 값으로 안전하게 추출
threshold_values = [v for v in threshold_stats.values() if v is not None]
if not threshold_values:
    raise ValueError(f"Threshold computation returned no valid value: {threshold_stats}")
threshold_distance = threshold_values[0]
print(f"Distance threshold (top {TOP_PERCENT}%): {threshold_distance:.4f}")

# 임계값을 통과한 픽셀만 단일 빨강으로 표시 (0인 픽셀은 마스크 처리되어 투명)
top_mask = min_distance.lte(threshold_distance)
similarity_layer = top_mask.updateMask(top_mask).clip(korea_geom)

# ==============================
# 파이프라인 6: 지도 레이어 구성
# ==============================
vis_params = {"palette": ["#CB181D"]}

# 한국 중심 좌표 기준으로 지도 초기화
centroid_pt = korea_geom.centroid(maxError=1000).coordinates().getInfo()
Map = geemap.Map(center=[centroid_pt[1], centroid_pt[0]], zoom=7)

# 전국 통합 레이어 (기본 ON)
Map.addLayer(similarity_layer, vis_params, "전국 산불 유사도", shown=True)

# 행정구역별 토글 레이어 (기본 OFF)
print("Adding per-region layers...")
region_info = kor_adm1.getInfo()
region_features_raw = region_info.get("features", [])
print(f"Region count: {len(region_features_raw)}")
for feat in region_features_raw:
    raw_name = feat["properties"].get("shapeName", "Unknown")
    display_name = to_korean_region_name(raw_name)
    region_geom = ee.Feature(feat).geometry()
    region_layer = similarity_layer.clip(region_geom)
    Map.addLayer(region_layer, vis_params, f"{display_name}", shown=False)

# 보조 레이어: 행정구역 경계, 산불 발생 지점
Map.addLayer(
    ee.Image().paint(kor_adm1, 1, 2),
    {"palette": ["#000000"]},
    "행정구역 경계",
    shown=True,
)
Map.addLayer(fire_fc, {"color": "#542788"}, "산불 발생 지점", shown=False)

Map.addLayerControl()
Map.to_html(OUTPUT_HTML)
print(f"Saved: {OUTPUT_HTML}")
