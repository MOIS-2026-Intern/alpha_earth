import csv
import json
import os

import ee


# 분석 스크립트 기준 경로를 고정해, 어디서 실행해도 입력/출력 파일을 안정적으로 찾는다.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(SCRIPT_DIR)


# .env 파일을 읽어 Earth Engine 프로젝트 ID와 분석 설정을 환경변수로 주입한다.
def load_env_file(path):
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


for env_path in [
    os.path.join(REPO_DIR, ".env"),
    os.path.join(SCRIPT_DIR, ".env"),
    os.path.abspath(".env"),
]:
    load_env_file(env_path)


# 입력 파일은 현재 실행 위치, 스크립트 폴더, source 폴더 순서로 탐색한다.
def resolve_input_path(path):
    if os.path.isabs(path):
        return path

    candidates = [
        os.path.abspath(path),
        os.path.join(SCRIPT_DIR, path),
        os.path.join(SCRIPT_DIR, "source", os.path.basename(path)),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return candidates[-1]


def resolve_output_path(path):
    if os.path.isabs(path):
        return path
    return os.path.join(SCRIPT_DIR, path)


# Earth Engine 객체나 tuple처럼 JSON 직렬화가 어려운 값을 저장 가능한 형태로 정리한다.
def write_json(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(to_json_safe(payload), f, ensure_ascii=False, indent=2)


def write_csv(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not rows:
        with open(path, "w", encoding="utf-8", newline="") as f:
            f.write("")
        return

    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)

    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value, ensure_ascii=False)
                    if isinstance(value, (dict, list))
                    else value
                    for key, value in row.items()
                }
            )


# 실험 결과 저장 전에 Python 기본 타입으로 변환한다.
def to_json_safe(value):
    if isinstance(value, dict):
        return {str(key): to_json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [to_json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def parse_number(value):
    cleaned = str(value or "").replace(",", "").strip()
    if not cleaned:
        return 0.0
    try:
        return float(cleaned)
    except ValueError:
        return 0.0


# 공개데이터의 한국어 자치구명을 geoBoundaries ADM2의 영어 shapeName과 연결한다.
SEOUL_GU_TO_ADM2_SHAPE = {
    "종로구": "Jongno-gu",
    "중구": "Jung-gu [Central District]",
    "용산구": "Yongsan-gu",
    "성동구": "Seongdong-gu",
    "광진구": "Gwangjin-gu",
    "동대문구": "Dongdaemun-gu",
    "중랑구": "Jungnang-gu",
    "성북구": "Seongbuk-gu",
    "강북구": "Gangbuk-gu",
    "도봉구": "Dobong-gu",
    "노원구": "Nowon-gu",
    "은평구": "Eunpyeong-gu",
    "서대문구": "Seodaemun-gu",
    "마포구": "Mapo-gu",
    "양천구": "Yangcheon-gu",
    "강서구": "Gangseo-gu",
    "구로구": "Guro-gu",
    "금천구": "Geumcheon-gu",
    "영등포구": "Yeongdeungpo-gu",
    "동작구": "Dongjak-gu",
    "관악구": "Gwanak-gu",
    "서초구": "Seocho-gu",
    "강남구": "Gangnam-gu",
    "송파구": "Songpa-gu",
    "강동구": "Gangdong-gu",
}
SEOUL_GU_ORDER = list(SEOUL_GU_TO_ADM2_SHAPE)
SEOUL_GU_NAMES = set(SEOUL_GU_TO_ADM2_SHAPE)


# CSV 주소/관리기관명에서 서울 자치구명을 표준 형태로 추출한다.
def normalize_seoul_gu_name(value):
    name = str(value or "").strip()
    if not name:
        return None
    if name == "중":
        return "중구"
    if not name.endswith("구") and f"{name}구" in SEOUL_GU_NAMES:
        return f"{name}구"
    return name if name in SEOUL_GU_NAMES else None


def extract_seoul_gu_from_text(*values):
    text = " ".join(str(value or "") for value in values)
    matches = [
        (text.find(gu_name), gu_name)
        for gu_name in SEOUL_GU_ORDER
        if gu_name in text
    ]
    return sorted(matches)[0][1] if matches else None


def read_csv_with_fallback(path, encodings=("utf-8-sig", "cp949")):
    last_error = None
    for encoding in encodings:
        try:
            with open(path, "r", encoding=encoding, newline="") as f:
                return list(csv.DictReader(f)), encoding
        except UnicodeDecodeError as error:
            last_error = error
    raise last_error


# 구별 원자료를 0~1 범위로 정규화해 서로 다른 단위의 인프라 지표를 모델 feature로 쓸 수 있게 한다.
def normalize_gu_stats(raw_stats_by_gu, metric_names):
    max_by_metric = {
        metric: max(
            (stats.get(metric, 0.0) for stats in raw_stats_by_gu.values()),
            default=0.0,
        )
        for metric in metric_names
    }
    normalized = {}
    for gu_name, raw_stats in raw_stats_by_gu.items():
        normalized[gu_name] = {}
        for metric in metric_names:
            max_value = max_by_metric[metric]
            normalized[gu_name][f"{metric}_norm"] = (
                raw_stats.get(metric, 0.0) / max_value if max_value else 0.0
            )
    return normalized, max_by_metric


# 배수펌프장 원자료를 자치구별 펌프장 수/용량/유역/유수지 통계로 요약한다.
def load_pump_station_gu_stats(csv_path, min_districts=5):
    summary = {
        "dataset": "pump_station_gu_stats",
        "path": csv_path,
        "exists": os.path.exists(csv_path),
        "used": False,
    }
    if not summary["exists"]:
        summary["reason"] = "file_missing"
        return {}, summary

    rows, encoding = read_csv_with_fallback(csv_path, encodings=("cp949", "utf-8-sig"))
    unique_stations = {}
    skipped_rows = 0
    for row in rows:
        gu_name = extract_seoul_gu_from_text(row.get("상세주소"), row.get("주소"))
        if gu_name is None:
            gu_name = normalize_seoul_gu_name(row.get("시설관리자"))
        if gu_name is None:
            skipped_rows += 1
            continue

        address = (row.get("상세주소") or row.get("주소") or "").strip()
        key = (gu_name, (row.get("시설물명") or "").strip(), address)
        # 같은 펌프장이 여러 행으로 반복될 수 있어 시설명/주소 단위로 중복을 묶는다.
        station_stats = unique_stations.setdefault(
            key,
            {
                "pump_station_count": 1.0,
                "pump_capacity_sum": 0.0,
                "pump_catchment_area_sum": 0.0,
                "pump_reservoir_capacity_sum": 0.0,
            },
        )
        station_stats["pump_capacity_sum"] = max(
            station_stats["pump_capacity_sum"],
            parse_number(row.get("배수장_최대배수량")),
        )
        station_stats["pump_catchment_area_sum"] = max(
            station_stats["pump_catchment_area_sum"],
            parse_number(row.get("유역면적")),
        )
        station_stats["pump_reservoir_capacity_sum"] = max(
            station_stats["pump_reservoir_capacity_sum"],
            parse_number(row.get("유수지용량")),
        )

    raw_stats_by_gu = {}
    for (gu_name, _, _), station_stats in unique_stations.items():
        gu_stats = raw_stats_by_gu.setdefault(
            gu_name,
            {
                "pump_station_count": 0.0,
                "pump_capacity_sum": 0.0,
                "pump_catchment_area_sum": 0.0,
                "pump_reservoir_capacity_sum": 0.0,
            },
        )
        for metric_name, metric_value in station_stats.items():
            gu_stats[metric_name] += metric_value

    metric_names = [
        "pump_station_count",
        "pump_capacity_sum",
        "pump_catchment_area_sum",
        "pump_reservoir_capacity_sum",
    ]
    normalized_stats, max_by_metric = normalize_gu_stats(raw_stats_by_gu, metric_names)
    summary.update(
        {
            "encoding": encoding,
            "raw_records": len(rows),
            "unique_stations": len(unique_stations),
            "district_count": len(raw_stats_by_gu),
            "skipped_rows": skipped_rows,
            "max_by_metric": max_by_metric,
        }
    )
    if summary["district_count"] < min_districts:
        summary["reason"] = "too_few_districts"
        return {}, summary

    summary["used"] = True
    return normalized_stats, summary


# 하수관로 수위 센서 자료를 자치구별 센서 수 feature로 요약한다.
def load_sewer_sensor_gu_stats(summary_csv_path, raw_csv_path, min_districts=5):
    summary = {
        "dataset": "sewer_level_sensor_gu_stats",
        "summary_path": summary_csv_path,
        "raw_path": raw_csv_path,
        "summary_exists": os.path.exists(summary_csv_path),
        "raw_exists": os.path.exists(raw_csv_path),
        "used": False,
    }

    raw_stats_by_gu = {}
    if summary["summary_exists"]:
        rows, encoding = read_csv_with_fallback(summary_csv_path)
        summary["encoding"] = encoding
        summary["source"] = "summary_csv"
        for row in rows:
            gu_name = normalize_seoul_gu_name(row.get("gu") or row.get("구분명"))
            if gu_name:
                raw_stats_by_gu[gu_name] = {
                    "sewer_sensor_count": parse_number(row.get("sensor_count"))
                }
    elif summary["raw_exists"]:
        sensor_ids_by_gu = {}
        last_error = None
        for encoding in ("cp949", "utf-8-sig"):
            try:
                with open(raw_csv_path, "r", encoding=encoding, newline="") as f:
                    reader = csv.DictReader(f)
                    id_field = reader.fieldnames[0] if reader.fieldnames else None
                    for row in reader:
                        gu_name = normalize_seoul_gu_name(row.get("구분명"))
                        sensor_id = str(row.get(id_field) or "").strip()
                        if gu_name and sensor_id:
                            sensor_ids_by_gu.setdefault(gu_name, set()).add(sensor_id)
                summary["encoding"] = encoding
                summary["source"] = "raw_csv"
                break
            except UnicodeDecodeError as error:
                last_error = error
        else:
            raise last_error
        raw_stats_by_gu = {
            gu_name: {"sewer_sensor_count": float(len(sensor_ids))}
            for gu_name, sensor_ids in sensor_ids_by_gu.items()
        }
    else:
        summary["reason"] = "file_missing"
        return {}, summary

    normalized_stats, max_by_metric = normalize_gu_stats(
        raw_stats_by_gu,
        ["sewer_sensor_count"],
    )
    summary.update(
        {
            "district_count": len(raw_stats_by_gu),
            "max_by_metric": max_by_metric,
        }
    )
    if summary["district_count"] < min_districts:
        summary["reason"] = "too_few_districts"
        return {}, summary

    summary["used"] = True
    return normalized_stats, summary


# 분석 설정: 30m sample scale, 5개 공간 fold, 양성/음성 sample 수와 buffer 거리를 정의한다.
PROJECT_ID = os.environ.get("EE_PROJECT_ID")
if not PROJECT_ID:
    raise ValueError("EE_PROJECT_ID is missing. Set it in .env or your shell.")

YEAR = int(os.environ.get("YEAR", "2024"))
ANALYSIS_SCALE = int(os.environ.get("ANALYSIS_SCALE", "30"))
SPATIAL_BLOCK_DEGREES = float(os.environ.get("SPATIAL_BLOCK_DEGREES", "0.015"))
SPATIAL_FOLDS = int(os.environ.get("SPATIAL_FOLDS", "5"))
POSITIVE_SAMPLE_POINTS = int(os.environ.get("POSITIVE_SAMPLE_POINTS", "200"))
NEGATIVE_POINTS = int(os.environ.get("NEGATIVE_POINTS", "200"))
POSITIVE_BUFFER_M = int(os.environ.get("POSITIVE_BUFFER_M", "60"))
NEGATIVE_BUFFER_M = int(os.environ.get("NEGATIVE_BUFFER_M", "300"))
OUTPUT_DIR = resolve_output_path(os.environ.get("OUTPUT_DIR", "outputs/analysis"))
REFERENCE_GEOJSON = resolve_input_path(
    os.environ.get("SEOUL_REFERENCE_GEOJSON", "source/seoul_flood_reference_points.geojson")
)
PUMP_STATION_CSV = resolve_input_path(
    os.environ.get("PUMP_STATION_CSV", "source/seoul_pump_stations.csv")
)
SEWER_SENSOR_GU_STATS_CSV = resolve_input_path(
    os.environ.get("SEWER_SENSOR_GU_STATS_CSV", "source/sewer_level_sensor_gu_stats.csv")
)
SEWER_LEVEL_SENSOR_CSV = resolve_input_path(
    os.environ.get("SEWER_LEVEL_SENSOR_CSV", "source/sewer_level_202605.csv")
)

METRICS_JSON = os.path.join(OUTPUT_DIR, "metrics.json")
CV_RESULTS_CSV = os.path.join(OUTPUT_DIR, "cv_results.csv")
FEATURE_IMPORTANCE_CSV = os.path.join(OUTPUT_DIR, "feature_importance.csv")

# Earth Engine 서버 계산을 시작하기 위해 프로젝트를 초기화한다.
ee.Initialize(project=PROJECT_ID)


# 경위도 기반 큰 공간 블록을 만들고, 각 블록을 5개 fold 중 하나로 배정한다.
# 랜덤 분할보다 공간적으로 가까운 sample이 train/validation에 섞이는 문제를 줄이기 위한 장치다.
def add_spatial_fold(fc):
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


# 모든 입력 raster와 sample을 서울 행정경계 안으로 제한한다.
def find_seoul_boundary():
    adm1 = ee.FeatureCollection("WM/geoLab/geoBoundaries/600/ADM1")
    seoul_fc = (
        adm1.filter(ee.Filter.eq("shapeGroup", "KOR"))
        .filter(
            ee.Filter.Or(
                ee.Filter.stringContains("shapeName", "Seoul"),
                ee.Filter.stringContains("shapeName", "SEOUL"),
                ee.Filter.stringContains("shapeName", "서울"),
            )
        )
    )
    if seoul_fc.size().getInfo() == 0:
        raise ValueError("Could not find Seoul boundary from geoBoundaries ADM1.")
    return seoul_fc, seoul_fc.geometry()


seoul_fc, seoul = find_seoul_boundary()


# 침수흔적도 기반 기준점을 label=1 양성 target으로 읽고, 서울 경계 안의 점만 분석에 사용한다.
def load_positive_points():
    with open(REFERENCE_GEOJSON, "r", encoding="utf-8") as f:
        geojson = json.load(f)

    features = [
        ee.Feature(
            ee.Geometry.Point(feature["geometry"]["coordinates"]),
            {**feature["properties"], "label": 1},
        )
        for feature in geojson["features"]
    ]
    all_points = add_spatial_fold(ee.FeatureCollection(features))
    points = all_points.filterBounds(seoul)
    return {
        "all_points": all_points,
        "points": points,
        "all_count": all_points.size().getInfo(),
        "analysis_count": points.size().getInfo(),
        "fold_histogram": points.aggregate_histogram("fold").getInfo(),
    }


positive_data = load_positive_points()
positive_points = positive_data["points"]
positive_geom = positive_points.geometry()


# 서울 전체 raster 픽셀에도 같은 fold 규칙을 적용해 train/validation 영역 마스크를 만들 수 있게 한다.
def make_spatial_fold_image():
    lonlat = ee.Image.pixelLonLat()
    block_x = lonlat.select("longitude").add(180).divide(SPATIAL_BLOCK_DEGREES).floor()
    block_y = lonlat.select("latitude").add(90).divide(SPATIAL_BLOCK_DEGREES).floor()
    return (
        block_x.multiply(73856093)
        .add(block_y.multiply(19349663))
        .mod(SPATIAL_FOLDS)
        .rename("spatial_fold")
        .clip(seoul)
    )


spatial_fold = make_spatial_fold_image()


# 자치구 단위 통계값을 ADM2 polygon에 붙인 뒤 raster band로 변환한다.
# 같은 자치구 안에서는 같은 값이므로, 세밀한 시설 영향권이 아니라 구 단위 배경 feature다.
def make_gu_stat_feature_images(stats_by_gu, dataset_name, dem_projection):
    band_names = sorted(
        {band for gu_stats in stats_by_gu.values() for band in gu_stats}
    )
    summary = {
        "dataset": dataset_name,
        "band_names": band_names,
        "district_count": len(stats_by_gu),
        "used": False,
    }
    if not band_names:
        summary["reason"] = "no_bands"
        return [], [], summary

    stats_by_shape_name = {}
    for gu_name, gu_stats in stats_by_gu.items():
        shape_name = SEOUL_GU_TO_ADM2_SHAPE.get(gu_name)
        if shape_name:
            stats_by_shape_name[shape_name] = gu_stats

    if not stats_by_shape_name:
        summary["reason"] = "no_mapped_districts"
        return [], [], summary

    default_props = {band_name: 0 for band_name in band_names}
    stats_dict = ee.Dictionary(stats_by_shape_name)
    adm2 = (
        ee.FeatureCollection("WM/geoLab/geoBoundaries/600/ADM2")
        .filter(ee.Filter.eq("shapeGroup", "KOR"))
        .filterBounds(seoul)
        .filter(ee.Filter.inList("shapeName", list(stats_by_shape_name.keys())))
    )

    def _set_gu_stats(feature):
        stats = ee.Dictionary(stats_dict.get(feature.get("shapeName"), default_props))
        return feature.setMulti(stats)

    gu_stats_fc = adm2.map(_set_gu_stats)
    matched_count = gu_stats_fc.size().getInfo()
    if matched_count == 0:
        summary["reason"] = "adm2_not_matched"
        return [], [], summary

    images = [
        gu_stats_fc.reduceToImage([band_name], ee.Reducer.first())
        .unmask(0)
        .rename(band_name)
        .clip(seoul)
        .reproject(crs=dem_projection, scale=ANALYSIS_SCALE)
        for band_name in band_names
    ]
    summary["used"] = True
    summary["matched_adm2_count"] = matched_count
    return images, band_names, summary


# 모델이 사용할 정적 feature 이미지를 만든다.
# 지형/수문/토지피복 feature와 선택적으로 자치구 단위 배수 인프라 feature를 합친다.
def build_static_features():
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
    dynamic_world = (
        ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1")
        .filterDate(f"{YEAR}-01-01", f"{YEAR + 1}-01-01")
        .filterBounds(seoul)
        .select(["built", "water"])
        .mean()
        .clip(seoul)
    )
    built = dynamic_world.select("built").rename("built")
    dw_water = dynamic_world.select("water").rename("dw_water")

    # 서울 내부 고도 범위에서 상대적으로 낮은 정도를 lowland feature로 만든다.
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

    drainage_summaries = []
    drainage_images = []
    drainage_bands = []
    # 배수펌프장과 하수관로 센서 통계가 충분히 있으면 구 단위 raster feature로 추가한다.
    for stats_by_gu, source_summary in [
        load_pump_station_gu_stats(PUMP_STATION_CSV),
        load_sewer_sensor_gu_stats(SEWER_SENSOR_GU_STATS_CSV, SEWER_LEVEL_SENSOR_CSV),
    ]:
        if source_summary["used"]:
            images, bands, image_summary = make_gu_stat_feature_images(
                stats_by_gu,
                source_summary["dataset"],
                dem.projection(),
            )
            source_summary["image_summary"] = image_summary
            if image_summary["used"]:
                drainage_images.extend(images)
                drainage_bands.extend(bands)
        drainage_summaries.append(source_summary)

    static_image = ee.Image.cat(
        [slope, hnd, log_upa, water_occ, built, lowland] + drainage_images
    )
    base_bands = ["slope", "hnd", "log_upa", "water_occ", "built", "lowland"]
    return {
        "image": static_image,
        "base_bands": base_bands,
        "drainage_bands": drainage_bands,
        "water_occ": water_occ,
        "dw_water": dw_water,
        "drainage_summaries": drainage_summaries,
    }


static_features = build_static_features()


# AlphaEarth/Satellite Embedding annual image를 서울 영역으로 가져온다.
# 원본 64차원 embedding은 이후 alpha_score 1개 feature로 축약된다.
def load_alphaearth():
    collection = (
        ee.ImageCollection("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL")
        .filterDate(f"{YEAR}-01-01", f"{YEAR + 1}-01-01")
        .filterBounds(seoul)
    )
    count = collection.size().getInfo()
    if count == 0:
        raise ValueError(f"AlphaEarth annual embeddings not found for Seoul in {YEAR}.")
    image = collection.mosaic().clip(seoul)
    band_names = image.bandNames().getInfo()
    return image, image.select(band_names).unmask(0), band_names, count


alpha_image, alpha_valid, alpha_bands, alpha_tile_count = load_alphaearth()


# 음성 후보 영역: 서울 안에서 물 영역과 침수 기준점 주변 buffer를 제외한 곳이다.
# 침수점 근처를 음성으로 잘못 뽑는 label noise를 줄이기 위해 negative buffer를 둔다.
def make_negative_mask(negative_buffer_m):
    positive_exclusion = ee.Image.constant(0).byte().paint(
        positive_geom.buffer(negative_buffer_m),
        1,
    )
    return (
        ee.Image.constant(1)
        .clip(seoul)
        .updateMask(static_features["water_occ"].lt(0.2))
        .updateMask(static_features["dw_water"].lt(0.25))
        .updateMask(positive_exclusion.eq(0))
        .rename("negative_mask")
    )


# 학습용 침수 기준점의 평균 AlphaEarth embedding과 각 위치 embedding 간 거리를 유사도 점수로 변환한다.
# validation fold 기준점은 평균 계산에 넣지 않아 데이터 누수를 줄인다.
def make_alpha_score(reference_fc):
    reference_geom = reference_fc.geometry()
    reference_mean = ee.Dictionary(
        alpha_image.clip(reference_geom)
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
    mean_image = reference_mean.toImage(alpha_bands).rename(alpha_bands)
    alpha_distance = (
        alpha_valid.subtract(mean_image)
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


fold_cache = {}


# 한 validation fold에 필요한 feature image, 양성 mask, train/validation 영역 mask를 준비한다.
def build_fold_inputs(validation_fold):
    if validation_fold in fold_cache:
        return fold_cache[validation_fold]

    train_positive = positive_points.filter(ee.Filter.neq("fold", validation_fold))
    valid_positive = positive_points.filter(ee.Filter.eq("fold", validation_fold))
    alpha_score = make_alpha_score(train_positive)
    feature_image = static_features["image"].addBands(alpha_score)
    fold_inputs = {
        "feature_image": feature_image,
        "train_positive": train_positive,
        "valid_positive": valid_positive,
        "train_area_mask": spatial_fold.neq(validation_fold),
        "valid_area_mask": spatial_fold.eq(validation_fold),
        # 침수 기준점 하나만 쓰지 않고 주변 60m를 양성 sample 후보 영역으로 확장한다.
        "positive_train_mask": ee.Image.constant(0).byte().paint(
            train_positive.map(lambda f: f.buffer(POSITIVE_BUFFER_M)),
            1,
        ),
        "positive_valid_mask": ee.Image.constant(0).byte().paint(
            valid_positive.map(lambda f: f.buffer(POSITIVE_BUFFER_M)),
            1,
        ),
    }
    fold_cache[validation_fold] = fold_inputs
    return fold_inputs


# Random Forest 입력 feature 목록이다. AlphaEarth는 원본 64개 band가 아니라 alpha_score 1개로 들어간다.
INPUT_BANDS = (
    static_features["base_bands"]
    + ["alpha_score"]
    + static_features["drainage_bands"]
)


# 지정된 train 또는 validation 영역에서 양성/음성 sample을 균형 있게 추출한다.
# ANALYSIS_SCALE 기본값 30m가 실제 sample 및 feature 추출 스케일이다.
def sample_split(feature_image, input_bands, positive_mask, negative_mask, area_mask, seed):
    split_positive = positive_mask.updateMask(area_mask)
    split_negative = negative_mask.updateMask(area_mask)
    label_image = (
        ee.Image.constant(0)
        .clip(seoul)
        .where(split_positive.unmask(0).eq(1), 1)
        .rename("label")
    )
    sampling_mask = split_negative.unmask(0).add(split_positive.unmask(0)).gt(0)
    split_image = (
        feature_image.select(input_bands)
        .addBands(label_image)
        .updateMask(sampling_mask)
    )
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


# Earth Engine의 Random Forest를 확률 출력 모드로 학습한다.
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


# validation sample에 모델을 적용하고 threshold 0.5 기준 혼동행렬을 만든다.
def evaluate_fc(fc, classifier):
    evaluated = fc.classify(classifier, "probability").map(
        lambda f: f.set("predicted", ee.Number(f.get("probability")).gte(0.5).int())
    )
    return evaluated.errorMatrix("label", "predicted")


# accuracy만으로는 침수 탐지 성능을 설명하기 부족해 여러 분류 지표를 직접 계산한다.
def safe_divide(numerator, denominator):
    return numerator / denominator if denominator else None


# score 순위 기반 ROC-AUC를 계산한다.
def auc_roc_from_scores(labels, scores):
    positive_count = sum(labels)
    negative_count = len(labels) - positive_count
    if positive_count == 0 or negative_count == 0:
        return None

    ranked = sorted(zip(scores, labels), key=lambda item: item[0])
    positive_rank_sum = 0.0
    index = 0
    while index < len(ranked):
        end = index + 1
        while end < len(ranked) and ranked[end][0] == ranked[index][0]:
            end += 1
        average_rank = (index + 1 + end) / 2
        positive_rank_sum += sum(label for _, label in ranked[index:end]) * average_rank
        index = end

    return (
        positive_rank_sum - positive_count * (positive_count + 1) / 2
    ) / (positive_count * negative_count)


# 양성 탐지 문제에서 중요한 PR-AUC/average precision을 계산한다.
def average_precision_from_scores(labels, scores):
    positive_count = sum(labels)
    if positive_count == 0:
        return None

    true_positive = 0
    false_positive = 0
    precision_at_positive = []
    for _, label in sorted(zip(scores, labels), key=lambda item: item[0], reverse=True):
        if label == 1:
            true_positive += 1
            precision_at_positive.append(true_positive / (true_positive + false_positive))
        else:
            false_positive += 1
    return sum(precision_at_positive) / positive_count


# threshold 0.5 기준의 confusion metric과 threshold-free metric을 함께 반환한다.
def binary_metrics(labels, scores, threshold=0.5):
    predictions = [1 if score >= threshold else 0 for score in scores]
    tp = sum(1 for y, p in zip(labels, predictions) if y == 1 and p == 1)
    tn = sum(1 for y, p in zip(labels, predictions) if y == 0 and p == 0)
    fp = sum(1 for y, p in zip(labels, predictions) if y == 0 and p == 1)
    fn = sum(1 for y, p in zip(labels, predictions) if y == 1 and p == 0)

    precision = safe_divide(tp, tp + fp)
    recall = safe_divide(tp, tp + fn)
    specificity = safe_divide(tn, tn + fp)
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and precision + recall > 0
        else None
    )
    return {
        "threshold": threshold,
        "sample_count": len(labels),
        "positive_count": sum(labels),
        "negative_count": len(labels) - sum(labels),
        "true_positive": tp,
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
        "accuracy": safe_divide(tp + tn, len(labels)),
        "precision": precision,
        "recall": recall,
        "sensitivity": recall,
        "specificity": specificity,
        "f1": f1,
        "balanced_accuracy": (
            (recall + specificity) / 2
            if recall is not None and specificity is not None
            else None
        ),
        "negative_predictive_value": safe_divide(tn, tn + fn),
        "false_positive_rate": 1 - specificity if specificity is not None else None,
        "false_negative_rate": 1 - recall if recall is not None else None,
        "roc_auc": auc_roc_from_scores(labels, scores),
        "pr_auc": average_precision_from_scores(labels, scores),
        "average_precision": average_precision_from_scores(labels, scores),
    }


# Earth Engine FeatureCollection 예측 결과를 Python 리스트로 가져와 metric을 계산한다.
def sample_metrics(sample_fc, classifier):
    evaluated = sample_fc.classify(classifier, "probability")
    features = evaluated.select(["label", "probability"]).getInfo()["features"]
    labels = [int(feature["properties"]["label"]) for feature in features]
    scores = [float(feature["properties"]["probability"]) for feature in features]
    return binary_metrics(labels, scores)


# Random Forest가 어떤 feature를 많이 사용했는지 fold별 중요도를 가져온다.
def classifier_importance(classifier, input_bands):
    explain_info = classifier.explain().getInfo()
    raw_importance = explain_info.get("importance", {})
    importance = {band: float(raw_importance.get(band, 0)) for band in input_bands}
    total = sum(importance.values())
    normalized = {
        band: (value / total if total else 0)
        for band, value in importance.items()
    }
    return importance, normalized


# 하나의 validation fold에 대해 sample 추출, RF 학습, validation 평가, feature importance 계산을 수행한다.
def run_fold(validation_fold):
    fold_inputs = build_fold_inputs(validation_fold)
    negative_mask = make_negative_mask(NEGATIVE_BUFFER_M)
    train_fc = sample_split(
        fold_inputs["feature_image"],
        INPUT_BANDS,
        fold_inputs["positive_train_mask"],
        negative_mask,
        fold_inputs["train_area_mask"],
        700 + validation_fold,
    )
    valid_fc = sample_split(
        fold_inputs["feature_image"],
        INPUT_BANDS,
        fold_inputs["positive_valid_mask"],
        negative_mask,
        fold_inputs["valid_area_mask"],
        1700 + validation_fold,
    )
    classifier = train_rf(train_fc, INPUT_BANDS)
    valid_confusion = evaluate_fc(valid_fc, classifier)
    importance, normalized_importance = classifier_importance(classifier, INPUT_BANDS)
    result = {
        "fold": validation_fold,
        "train_positive_count": fold_inputs["train_positive"].size().getInfo(),
        "valid_positive_count": fold_inputs["valid_positive"].size().getInfo(),
        "train_sample_count": train_fc.size().getInfo(),
        "valid_sample_count": valid_fc.size().getInfo(),
        "train_label_histogram": train_fc.aggregate_histogram("label").getInfo(),
        "valid_label_histogram": valid_fc.aggregate_histogram("label").getInfo(),
        "valid_confusion": valid_confusion.getInfo(),
        "valid_accuracy": valid_confusion.accuracy().getInfo(),
        "valid_kappa": valid_confusion.kappa().getInfo(),
        "valid_metrics": sample_metrics(valid_fc, classifier),
        "importance": importance,
        "importance_normalized": normalized_importance,
    }
    print(
        {
            "fold": result["fold"],
            "accuracy": result["valid_accuracy"],
            "kappa": result["valid_kappa"],
            "roc_auc": result["valid_metrics"]["roc_auc"],
            "pr_auc": result["valid_metrics"]["pr_auc"],
            "recall": result["valid_metrics"]["recall"],
            "f1": result["valid_metrics"]["f1"],
        }
    )
    return result


# fold별 결과를 평균/표준편차로 요약하기 위한 작은 통계 유틸리티다.
def mean(values):
    return sum(values) / max(len(values), 1)


def sample_std(values):
    if len(values) < 2:
        return 0
    avg = mean(values)
    return (sum((value - avg) ** 2 for value in values) / (len(values) - 1)) ** 0.5


# 최종 validation summary에 포함할 metric 목록이다.
METRIC_KEYS = [
    "accuracy",
    "kappa",
    "roc_auc",
    "pr_auc",
    "average_precision",
    "precision",
    "recall",
    "specificity",
    "f1",
    "balanced_accuracy",
    "negative_predictive_value",
    "false_positive_rate",
    "false_negative_rate",
]


# 5개 fold의 validation 결과를 평균과 표준편차로 요약한다.
def summarize_cv(rows):
    values_by_metric = {
        "accuracy": [row["valid_accuracy"] for row in rows],
        "kappa": [row["valid_kappa"] for row in rows],
    }
    for metric_key in METRIC_KEYS:
        if metric_key in values_by_metric:
            continue
        values_by_metric[metric_key] = [
            row["valid_metrics"][metric_key]
            for row in rows
            if row["valid_metrics"].get(metric_key) is not None
        ]

    summary = {}
    for metric_key, values in values_by_metric.items():
        if values:
            summary[f"{metric_key}_mean"] = mean(values)
            summary[f"{metric_key}_std"] = sample_std(values)
    return summary


# fold별 normalized feature importance를 평균/표준편차로 요약한다.
def summarize_importance(rows):
    summary = []
    for band in INPUT_BANDS:
        values = [
            row["importance_normalized"].get(band, 0)
            for row in rows
        ]
        summary.append(
            {
                "feature": band,
                "importance_mean": mean(values),
                "importance_std": sample_std(values),
            }
        )
    return sorted(summary, key=lambda row: row["importance_mean"], reverse=True)


# CSV/JSON 저장에 필요한 fold 결과만 남겨 출력 파일을 간결하게 만든다.
def sanitize_fold_result(row):
    return {
        "fold": row["fold"],
        "train_positive_count": row["train_positive_count"],
        "valid_positive_count": row["valid_positive_count"],
        "train_sample_count": row["train_sample_count"],
        "valid_sample_count": row["valid_sample_count"],
        "train_label_histogram": row["train_label_histogram"],
        "valid_label_histogram": row["valid_label_histogram"],
        "valid_confusion": row["valid_confusion"],
        "valid_accuracy": row["valid_accuracy"],
        "valid_kappa": row["valid_kappa"],
        "valid_metrics": row["valid_metrics"],
    }


# 전체 실행 진입점: 5-fold 공간 검증을 돌리고 metric/importance/output 경로를 저장한다.
def main():
    print("Analysis model bands:", INPUT_BANDS)
    print("Positive fold histogram:", positive_data["fold_histogram"])
    print("AlphaEarth tiles:", alpha_tile_count)

    cv_rows = [run_fold(fold) for fold in range(SPATIAL_FOLDS)]
    cv_summary = summarize_cv(cv_rows)
    importance_summary = summarize_importance(cv_rows)
    cv_result_rows = [sanitize_fold_result(row) for row in cv_rows]

    metrics_payload = {
        "config": {
            "year": YEAR,
            "analysis_scale": ANALYSIS_SCALE,
            "spatial_block_degrees": SPATIAL_BLOCK_DEGREES,
            "spatial_folds": SPATIAL_FOLDS,
            "positive_sample_points": POSITIVE_SAMPLE_POINTS,
            "negative_points": NEGATIVE_POINTS,
            "positive_buffer_m": POSITIVE_BUFFER_M,
            "negative_buffer_m": NEGATIVE_BUFFER_M,
            "reference_geojson": REFERENCE_GEOJSON,
            "pump_station_csv": PUMP_STATION_CSV,
            "sewer_sensor_gu_stats_csv": SEWER_SENSOR_GU_STATS_CSV,
            "sewer_level_sensor_csv": SEWER_LEVEL_SENSOR_CSV,
        },
        "data": {
            "reference_points_total": positive_data["all_count"],
            "reference_points_in_analysis_boundary": positive_data["analysis_count"],
            "positive_fold_histogram": positive_data["fold_histogram"],
            "drainage_infra": static_features["drainage_summaries"],
            "alphaearth_tiles": alpha_tile_count,
        },
        "model": {
            "name": "Hybrid-plus-drainage-gu-stats",
            "bands": INPUT_BANDS,
        },
        "validation": {
            "summary": cv_summary,
            "fold_results": cv_result_rows,
        },
        "feature_importance": importance_summary,
        "outputs": {
            "output_dir": OUTPUT_DIR,
            "metrics_json": METRICS_JSON,
            "cv_results_csv": CV_RESULTS_CSV,
            "feature_importance_csv": FEATURE_IMPORTANCE_CSV,
        },
    }

    write_json(METRICS_JSON, metrics_payload)
    write_csv(CV_RESULTS_CSV, cv_result_rows)
    write_csv(FEATURE_IMPORTANCE_CSV, importance_summary)
    print("Validation summary:", cv_summary)
    print("Saved:", metrics_payload["outputs"])


if __name__ == "__main__":
    main()
