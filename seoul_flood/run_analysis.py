import csv
import json
import os
import re
import sys
import tempfile
import zipfile

import ee
import shapefile

try:
    import optuna
    OPTUNA_IMPORT_ERROR = None
except ImportError as error:
    optuna = None
    OPTUNA_IMPORT_ERROR = error


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


def parse_int_list(raw_value):
    return [
        int(value.strip())
        for value in raw_value.split(",")
        if value.strip()
    ]


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
HOTSPOT_EVAL_PERCENTILES = parse_int_list(
    os.environ.get("HOTSPOT_EVAL_PERCENTILES", "80,90,95")
)
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
DEFAULT_OFFICIAL_FLOOD_SHP_FREQ_ROOT_DIR = os.path.join(
    SCRIPT_DIR,
    "source",
    "official_city_flood_by_frequency",
)
DEFAULT_OFFICIAL_FLOOD_SHP_ZIP_DIR = os.path.join(
    SCRIPT_DIR,
    "source",
    "official_city_flood",
)
OFFICIAL_FLOOD_SHP_FREQ_ROOT_DIR = os.environ.get(
    "OFFICIAL_FLOOD_SHP_FREQ_ROOT_DIR",
    "",
).strip()
if OFFICIAL_FLOOD_SHP_FREQ_ROOT_DIR:
    OFFICIAL_FLOOD_SHP_FREQ_ROOT_DIR = resolve_input_path(
        OFFICIAL_FLOOD_SHP_FREQ_ROOT_DIR
    )
elif os.path.isdir(DEFAULT_OFFICIAL_FLOOD_SHP_FREQ_ROOT_DIR):
    OFFICIAL_FLOOD_SHP_FREQ_ROOT_DIR = DEFAULT_OFFICIAL_FLOOD_SHP_FREQ_ROOT_DIR
OFFICIAL_FLOOD_SHP_ZIP_DIR = os.environ.get("OFFICIAL_FLOOD_SHP_ZIP_DIR", "").strip()
if OFFICIAL_FLOOD_SHP_ZIP_DIR:
    OFFICIAL_FLOOD_SHP_ZIP_DIR = resolve_input_path(OFFICIAL_FLOOD_SHP_ZIP_DIR)
elif os.path.isdir(DEFAULT_OFFICIAL_FLOOD_SHP_ZIP_DIR):
    OFFICIAL_FLOOD_SHP_ZIP_DIR = DEFAULT_OFFICIAL_FLOOD_SHP_ZIP_DIR
OFFICIAL_FLOOD_SHP_PROJ = os.environ.get("OFFICIAL_FLOOD_SHP_PROJ", "EPSG:5186")
OFFICIAL_FLOOD_SHP_ENCODING = os.environ.get("OFFICIAL_FLOOD_SHP_ENCODING", "cp949")
OFFICIAL_FLOOD_SHP_SIMPLIFY_M = float(
    os.environ.get("OFFICIAL_FLOOD_SHP_SIMPLIFY_M", "30")
)
RUN_EXTERNAL_VALIDATION = os.environ.get(
    "RUN_EXTERNAL_VALIDATION",
    "1" if (OFFICIAL_FLOOD_SHP_FREQ_ROOT_DIR or OFFICIAL_FLOOD_SHP_ZIP_DIR) else "0",
) == "1"
EXTERNAL_VALIDATION_PERCENTILES = parse_int_list(
    os.environ.get("EXTERNAL_VALIDATION_PERCENTILES", "80,90,95")
)

METRICS_JSON = os.path.join(OUTPUT_DIR, "metrics.json")
CV_RESULTS_CSV = os.path.join(OUTPUT_DIR, "cv_results.csv")
FEATURE_IMPORTANCE_CSV = os.path.join(OUTPUT_DIR, "feature_importance.csv")
TOPK_SUMMARY_CSV = os.path.join(OUTPUT_DIR, "topk_summary.csv")
EXTERNAL_VALIDATION_CSV = os.path.join(OUTPUT_DIR, "external_validation.csv")
HYPERPARAMETER_TUNING_CSV = os.path.join(
    OUTPUT_DIR,
    "hyperparameter_tuning.csv",
)

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


# 모델 입력 feature 목록이다. AlphaEarth는 원본 64개 band가 아니라 alpha_score 1개로 들어간다.
INPUT_BANDS = (
    static_features["base_bands"]
    + ["alpha_score"]
    + static_features["drainage_bands"]
)


FINAL_MODEL_CONFIG = {
    "name": "GradientTreeBoostSoftVoting",
    "family": "gradient_tree_boost_soft_voting",
    "params": {
        "numberOfTrees": int(os.environ.get("GTB_NUMBER_OF_TREES", "100")),
        "shrinkage": float(os.environ.get("GTB_SHRINKAGE", "0.05")),
        "samplingRate": float(os.environ.get("GTB_SAMPLING_RATE", "0.7")),
        "maxNodes": int(os.environ.get("GTB_MAX_NODES", "32")),
        "seeds": parse_int_list(os.environ.get("GTB_VOTING_SEEDS", "13,23,37")),
    },
}
if not FINAL_MODEL_CONFIG["params"]["seeds"]:
    raise ValueError("GTB_VOTING_SEEDS must contain at least one integer seed.")

RUN_HYPERPARAMETER_TUNING = os.environ.get(
    "RUN_HYPERPARAMETER_TUNING",
    "0",
) == "1"
GTB_TUNING_FOLDS = [
    fold
    for fold in parse_int_list(
        os.environ.get(
            "GTB_TUNING_FOLDS",
            ",".join(str(fold) for fold in range(SPATIAL_FOLDS)),
        )
    )
    if 0 <= fold < SPATIAL_FOLDS
]
GTB_TUNING_SEEDS = parse_int_list(
    os.environ.get(
        "GTB_TUNING_SEEDS",
        str(FINAL_MODEL_CONFIG["params"]["seeds"][0]),
    )
)
GTB_OPTUNA_TRIALS = int(os.environ.get("GTB_OPTUNA_TRIALS", "20"))
GTB_OPTUNA_RANDOM_SEED = int(
    os.environ.get("GTB_OPTUNA_RANDOM_SEED", "42")
)
GTB_OPTUNA_STARTUP_TRIALS = int(
    os.environ.get("GTB_OPTUNA_STARTUP_TRIALS", "5")
)
GTB_OPTUNA_PRUNING = os.environ.get("GTB_OPTUNA_PRUNING", "1") == "1"
GTB_OPTUNA_WARMUP_FOLDS = int(
    os.environ.get("GTB_OPTUNA_WARMUP_FOLDS", "2")
)
GTB_OPTUNA_TREES_MIN = int(os.environ.get("GTB_OPTUNA_TREES_MIN", "50"))
GTB_OPTUNA_TREES_MAX = int(os.environ.get("GTB_OPTUNA_TREES_MAX", "300"))
GTB_OPTUNA_TREES_STEP = int(os.environ.get("GTB_OPTUNA_TREES_STEP", "25"))
GTB_OPTUNA_SHRINKAGE_MIN = float(
    os.environ.get("GTB_OPTUNA_SHRINKAGE_MIN", "0.01")
)
GTB_OPTUNA_SHRINKAGE_MAX = float(
    os.environ.get("GTB_OPTUNA_SHRINKAGE_MAX", "0.15")
)
GTB_OPTUNA_SAMPLING_RATE_MIN = float(
    os.environ.get("GTB_OPTUNA_SAMPLING_RATE_MIN", "0.5")
)
GTB_OPTUNA_SAMPLING_RATE_MAX = float(
    os.environ.get("GTB_OPTUNA_SAMPLING_RATE_MAX", "1.0")
)
GTB_OPTUNA_MAX_NODES_CHOICES = parse_int_list(
    os.environ.get("GTB_OPTUNA_MAX_NODES_CHOICES", "8,16,32,64")
)
GTB_TUNING_METRIC = os.environ.get(
    "GTB_TUNING_METRIC",
    "pr_auc_mean",
).strip()
GTB_TUNING_ALLOWED_METRICS = {
    "accuracy_mean",
    "balanced_accuracy_mean",
    "f1_mean",
    "kappa_mean",
    "pr_auc_mean",
    "precision_mean",
    "recall_mean",
    "roc_auc_mean",
}
if RUN_HYPERPARAMETER_TUNING:
    tuning_lists = [
        GTB_TUNING_FOLDS,
        GTB_TUNING_SEEDS,
        GTB_OPTUNA_MAX_NODES_CHOICES,
    ]
    if any(not values for values in tuning_lists):
        raise ValueError("All GTB tuning lists must contain at least one value.")
    if optuna is None:
        raise ImportError(
            "Optuna is required for hyperparameter tuning. "
            "Install dependencies with: pip install -r requirements.txt"
        ) from OPTUNA_IMPORT_ERROR
    if GTB_TUNING_METRIC not in GTB_TUNING_ALLOWED_METRICS:
        raise ValueError(
            "GTB_TUNING_METRIC must be one of: "
            + ", ".join(sorted(GTB_TUNING_ALLOWED_METRICS))
        )
    if GTB_OPTUNA_TRIALS <= 0:
        raise ValueError("GTB_OPTUNA_TRIALS must be positive.")
    if GTB_OPTUNA_STARTUP_TRIALS < 0:
        raise ValueError("GTB_OPTUNA_STARTUP_TRIALS must be zero or positive.")
    if GTB_OPTUNA_WARMUP_FOLDS < 0:
        raise ValueError("GTB_OPTUNA_WARMUP_FOLDS must be zero or positive.")
    if (
        GTB_OPTUNA_TREES_MIN <= 0
        or GTB_OPTUNA_TREES_MAX < GTB_OPTUNA_TREES_MIN
        or GTB_OPTUNA_TREES_STEP <= 0
    ):
        raise ValueError("Invalid Optuna numberOfTrees range.")
    if not (
        0 < GTB_OPTUNA_SHRINKAGE_MIN <= GTB_OPTUNA_SHRINKAGE_MAX
    ):
        raise ValueError("Invalid Optuna shrinkage range.")
    if not (
        0 < GTB_OPTUNA_SAMPLING_RATE_MIN
        <= GTB_OPTUNA_SAMPLING_RATE_MAX
        <= 1
    ):
        raise ValueError("Invalid Optuna samplingRate range.")
    if any(value <= 1 for value in GTB_OPTUNA_MAX_NODES_CHOICES):
        raise ValueError("GTB_OPTUNA_MAX_NODES_CHOICES values must exceed 1.")


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


# 단일 Earth Engine Gradient Tree Boosting 모델을 확률 출력 모드로 학습한다.
def train_gtb_classifier(train_fc, input_bands, seed, model_params=None):
    params = model_params or FINAL_MODEL_CONFIG["params"]
    return (
        ee.Classifier.smileGradientTreeBoost(
            numberOfTrees=params["numberOfTrees"],
            shrinkage=params["shrinkage"],
            samplingRate=params["samplingRate"],
            maxNodes=params["maxNodes"],
            seed=seed,
        )
        .setOutputMode("PROBABILITY")
        .train(train_fc, "label", input_bands)
    )


# 같은 GTB 구조를 seed만 다르게 여러 번 학습해 soft voting ensemble로 사용한다.
def train_classifier(train_fc, input_bands, model_params=None, seeds=None):
    params = model_params or FINAL_MODEL_CONFIG["params"]
    classifier_seeds = seeds or params["seeds"]
    return [
        train_gtb_classifier(train_fc, input_bands, seed, params)
        for seed in classifier_seeds
    ]


def as_classifier_list(classifiers):
    return list(classifiers) if isinstance(classifiers, (list, tuple)) else [classifiers]


def classify_fc_with_voting(fc, classifiers, output_name="probability"):
    classifiers = as_classifier_list(classifiers)
    classified = fc
    probability_names = []
    for index, classifier in enumerate(classifiers):
        probability_name = f"{output_name}_{index}"
        classified = classified.classify(classifier, probability_name)
        probability_names.append(probability_name)

    def _set_voted_probability(feature):
        probability_sum = ee.Number(0)
        for probability_name in probability_names:
            probability_sum = probability_sum.add(ee.Number(feature.get(probability_name)))
        return feature.set(output_name, probability_sum.divide(len(probability_names)))

    return classified.map(_set_voted_probability)


def classify_image_with_voting(feature_image, input_bands, classifiers, output_name="flood_prob"):
    classifiers = as_classifier_list(classifiers)
    probability_images = [
        feature_image
        .select(input_bands)
        .classify(classifier, f"{output_name}_{index}")
        .rename(f"{output_name}_{index}")
        for index, classifier in enumerate(classifiers)
    ]
    if len(probability_images) == 1:
        return probability_images[0].rename(output_name)
    return ee.Image.cat(probability_images).reduce(ee.Reducer.mean()).rename(output_name)


# validation sample에 voting 모델을 적용하고 threshold 0.5 기준 혼동행렬을 만든다.
def evaluate_fc(fc, classifier):
    evaluated = classify_fc_with_voting(fc, classifier, "probability").map(
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
    evaluated = classify_fc_with_voting(sample_fc, classifier, "probability")
    features = evaluated.select(["label", "probability"]).getInfo()["features"]
    labels = [int(feature["properties"]["label"]) for feature in features]
    scores = [float(feature["properties"]["probability"]) for feature in features]
    return binary_metrics(labels, scores)


def compute_hotspot_metrics(probability, validation_points):
    """위험도 상위 20/10/5% 영역이 검증 침수 기준점을 얼마나 포함하는지 계산한다."""
    scored_points = probability.sampleRegions(
        collection=validation_points,
        scale=ANALYSIS_SCALE,
        geometries=False,
        tileScale=4,
    ).filter(ee.Filter.notNull(["flood_prob"]))
    valid_count = validation_points.size()
    sampled_count = scored_points.size()
    excluded_count = valid_count.subtract(sampled_count)
    analysis_area = (
        ee.Image.pixelArea()
        .divide(1e6)
        .updateMask(probability.mask())
        .reduceRegion(
            reducer=ee.Reducer.sum(),
            geometry=seoul,
            scale=ANALYSIS_SCALE,
            maxPixels=1e9,
        )
        .get("area")
    )
    analysis_area = ee.Number(analysis_area)

    metric_features = []
    for percentile in HOTSPOT_EVAL_PERCENTILES:
        threshold = ee.Number(
            probability.reduceRegion(
                reducer=ee.Reducer.percentile([percentile]),
                geometry=seoul,
                scale=ANALYSIS_SCALE,
                maxPixels=1e9,
            ).get("flood_prob")
        )
        hotspots = probability.gte(threshold)
        hit_count = scored_points.filter(ee.Filter.gte("flood_prob", threshold)).size()
        point_recall_value = ee.Number(
            ee.Algorithms.If(
                sampled_count.gt(0),
                ee.Number(hit_count).divide(sampled_count),
                0,
            )
        )
        all_point_recall_value = ee.Number(
            ee.Algorithms.If(
                valid_count.gt(0),
                ee.Number(hit_count).divide(valid_count),
                0,
            )
        )
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
            .get("area")
        )
        hotspot_area = ee.Number(hotspot_area)
        hotspot_area_share = ee.Number(
            ee.Algorithms.If(
                analysis_area.gt(0),
                hotspot_area.divide(analysis_area),
                0,
            )
        )
        lift = ee.Algorithms.If(
            hotspot_area_share.gt(0),
            point_recall_value.divide(hotspot_area_share),
            None,
        )
        metric_features.append(
            ee.Feature(
                None,
                {
                    "percentile": percentile,
                    "top_percent": 100 - percentile,
                    "threshold": threshold,
                    "valid_positive_points": valid_count,
                    "sampled_positive_points": sampled_count,
                    "excluded_positive_points": excluded_count,
                    "sample_coverage": ee.Algorithms.If(
                        valid_count.gt(0),
                        ee.Number(sampled_count).divide(valid_count),
                        None,
                    ),
                    "hit_points": hit_count,
                    "evaluated_point_recall": ee.Algorithms.If(
                        sampled_count.gt(0),
                        point_recall_value,
                        None,
                    ),
                    "all_point_recall": ee.Algorithms.If(
                        valid_count.gt(0),
                        all_point_recall_value,
                        None,
                    ),
                    "evaluated_point_lift": lift,
                    "hotspot_area_share": hotspot_area_share,
                    "analysis_area_km2": analysis_area,
                    "hotspot_area_km2": hotspot_area,
                },
            )
        )

    metrics = ee.FeatureCollection(metric_features).getInfo()["features"]
    return [feature["properties"] for feature in metrics]


# voting에 참여한 모델들이 어떤 feature를 많이 사용했는지 평균 중요도를 가져온다.
def classifier_importance(classifier, input_bands):
    classifiers = as_classifier_list(classifier)
    importance_rows = []
    normalized_rows = []
    for index, single_classifier in enumerate(classifiers):
        try:
            explain_info = single_classifier.explain().getInfo()
        except Exception as error:
            print(f"Feature importance unavailable for voter {index}: {error}")
            explain_info = {}
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
        importance_rows.append(importance)
        normalized_rows.append(normalized)

    importance_mean = {
        band: sum(row.get(band, 0) for row in importance_rows) / len(importance_rows)
        for band in input_bands
    }
    normalized_mean = {
        band: sum(row.get(band, 0) for row in normalized_rows) / len(normalized_rows)
        for band in input_bands
    }
    return importance_mean, normalized_mean


fold_sample_cache = {}


def get_fold_samples(validation_fold):
    if validation_fold in fold_sample_cache:
        return fold_sample_cache[validation_fold]

    fold_inputs = build_fold_inputs(validation_fold)
    negative_mask = make_negative_mask(NEGATIVE_BUFFER_M)
    samples = {
        "fold_inputs": fold_inputs,
        "train_fc": sample_split(
            fold_inputs["feature_image"],
            INPUT_BANDS,
            fold_inputs["positive_train_mask"],
            negative_mask,
            fold_inputs["train_area_mask"],
            700 + validation_fold,
        ),
        "valid_fc": sample_split(
            fold_inputs["feature_image"],
            INPUT_BANDS,
            fold_inputs["positive_valid_mask"],
            negative_mask,
            fold_inputs["valid_area_mask"],
            1700 + validation_fold,
        ),
    }
    fold_sample_cache[validation_fold] = samples
    return samples


def kappa_from_binary_metrics(metrics):
    sample_count = metrics["sample_count"]
    if not sample_count:
        return None

    tp = metrics["true_positive"]
    tn = metrics["true_negative"]
    fp = metrics["false_positive"]
    fn = metrics["false_negative"]
    actual_negative = tn + fp
    actual_positive = tp + fn
    predicted_negative = tn + fn
    predicted_positive = tp + fp
    expected_accuracy = (
        actual_negative * predicted_negative
        + actual_positive * predicted_positive
    ) / (sample_count * sample_count)
    if expected_accuracy == 1:
        return 0.0
    return (metrics["accuracy"] - expected_accuracy) / (1 - expected_accuracy)


def run_tuning_fold(validation_fold, model_params):
    samples = get_fold_samples(validation_fold)
    classifier = train_classifier(
        samples["train_fc"],
        INPUT_BANDS,
        model_params=model_params,
        seeds=GTB_TUNING_SEEDS,
    )
    valid_metrics = sample_metrics(samples["valid_fc"], classifier)
    return {
        "fold": validation_fold,
        "valid_accuracy": valid_metrics["accuracy"],
        "valid_kappa": kappa_from_binary_metrics(valid_metrics),
        "valid_metrics": valid_metrics,
    }


def suggest_gtb_params(trial):
    return {
        "numberOfTrees": trial.suggest_int(
            "numberOfTrees",
            GTB_OPTUNA_TREES_MIN,
            GTB_OPTUNA_TREES_MAX,
            step=GTB_OPTUNA_TREES_STEP,
        ),
        "shrinkage": trial.suggest_float(
            "shrinkage",
            GTB_OPTUNA_SHRINKAGE_MIN,
            GTB_OPTUNA_SHRINKAGE_MAX,
            log=True,
        ),
        "samplingRate": trial.suggest_float(
            "samplingRate",
            GTB_OPTUNA_SAMPLING_RATE_MIN,
            GTB_OPTUNA_SAMPLING_RATE_MAX,
        ),
        "maxNodes": trial.suggest_categorical(
            "maxNodes",
            GTB_OPTUNA_MAX_NODES_CHOICES,
        ),
    }


def tuning_sort_key(row):
    def _value(key, default=float("-inf")):
        value = row.get(key)
        return default if value is None else value

    metric_std = row.get(GTB_TUNING_METRIC.replace("_mean", "_std"))
    return (
        _value(GTB_TUNING_METRIC),
        -metric_std if metric_std is not None else float("-inf"),
        _value("pr_auc_mean"),
        _value("roc_auc_mean"),
        _value("kappa_mean"),
        _value("accuracy_mean"),
    )


def run_hyperparameter_tuning():
    sampler = optuna.samplers.TPESampler(
        seed=GTB_OPTUNA_RANDOM_SEED,
        n_startup_trials=GTB_OPTUNA_STARTUP_TRIALS,
    )
    pruner = (
        optuna.pruners.MedianPruner(
            n_startup_trials=GTB_OPTUNA_STARTUP_TRIALS,
            n_warmup_steps=GTB_OPTUNA_WARMUP_FOLDS,
        )
        if GTB_OPTUNA_PRUNING
        else optuna.pruners.NopPruner()
    )
    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        study_name="seoul_flood_gtb",
    )
    print(
        "Optuna hyperparameter tuning:",
        {
            "trials": GTB_OPTUNA_TRIALS,
            "folds": GTB_TUNING_FOLDS,
            "seeds": GTB_TUNING_SEEDS,
            "selection_metric": GTB_TUNING_METRIC,
            "sampler": "TPESampler",
            "pruning": GTB_OPTUNA_PRUNING,
        },
    )

    def objective(trial):
        model_params = suggest_gtb_params(trial)
        print(
            f"Optuna trial {trial.number + 1}/{GTB_OPTUNA_TRIALS}:",
            model_params,
        )
        fold_rows = []
        for step, fold in enumerate(GTB_TUNING_FOLDS):
            fold_rows.append(run_tuning_fold(fold, model_params))
            partial_summary = summarize_cv(fold_rows)
            partial_score = partial_summary.get(GTB_TUNING_METRIC)
            if partial_score is None:
                continue
            trial.report(partial_score, step=step)
            trial.set_user_attr("summary", to_json_safe(partial_summary))
            trial.set_user_attr("evaluated_folds", step + 1)
            if GTB_OPTUNA_PRUNING and trial.should_prune():
                print(
                    "Pruned Optuna trial:",
                    {
                        "trial": trial.number,
                        "evaluated_folds": step + 1,
                        GTB_TUNING_METRIC: partial_score,
                    },
                )
                raise optuna.TrialPruned()

        summary = summarize_cv(fold_rows)
        score = summary.get(GTB_TUNING_METRIC)
        if score is None:
            raise ValueError(
                f"Tuning metric {GTB_TUNING_METRIC} was not produced."
            )
        trial.set_user_attr("summary", to_json_safe(summary))
        trial.set_user_attr("evaluated_folds", len(fold_rows))
        print(
            {
                "trial": trial.number,
                **model_params,
                GTB_TUNING_METRIC: score,
                "kappa_mean": summary.get("kappa_mean"),
                "roc_auc_mean": summary.get("roc_auc_mean"),
            }
        )
        return score

    study.optimize(objective, n_trials=GTB_OPTUNA_TRIALS)

    rows = []
    for trial in study.trials:
        summary = trial.user_attrs.get("summary", {})
        duration_seconds = (
            trial.duration.total_seconds()
            if trial.duration is not None
            else None
        )
        rows.append(
            {
                "config_id": trial.number + 1,
                "trial_number": trial.number,
                "state": trial.state.name.lower(),
                **trial.params,
                "objective_value": trial.value,
                "selection_metric": GTB_TUNING_METRIC,
                "folds": GTB_TUNING_FOLDS,
                "evaluated_folds": trial.user_attrs.get("evaluated_folds", 0),
                "seeds": GTB_TUNING_SEEDS,
                "classifier_count_per_fold": len(GTB_TUNING_SEEDS),
                "duration_seconds": duration_seconds,
                **summary,
            }
        )

    completed_rows = [
        row for row in rows
        if row["state"] == "complete"
    ]
    ranked_rows = sorted(completed_rows, key=tuning_sort_key, reverse=True)
    rank_by_trial = {
        row["trial_number"]: rank
        for rank, row in enumerate(ranked_rows, start=1)
    }
    for row in rows:
        row["rank"] = rank_by_trial.get(row["trial_number"])
        row["selected"] = row["trial_number"] == study.best_trial.number
    rows = sorted(
        rows,
        key=lambda row: (
            row["rank"] is None,
            row["rank"] if row["rank"] is not None else row["trial_number"],
        ),
    )

    best_row = next(row for row in rows if row["selected"])
    selected_params = {
        "numberOfTrees": best_row["numberOfTrees"],
        "shrinkage": best_row["shrinkage"],
        "samplingRate": best_row["samplingRate"],
        "maxNodes": best_row["maxNodes"],
    }
    FINAL_MODEL_CONFIG["params"].update(selected_params)
    print(
        "Selected hyperparameters:",
        {
            **selected_params,
            GTB_TUNING_METRIC: best_row[GTB_TUNING_METRIC],
        },
    )
    return rows, {
        "used": True,
        "method": "optuna",
        "sampler": "TPESampler",
        "pruner": "MedianPruner" if GTB_OPTUNA_PRUNING else "NopPruner",
        "trial_count": len(study.trials),
        "completed_trial_count": len(completed_rows),
        "pruned_trial_count": sum(
            row["state"] == "pruned"
            for row in rows
        ),
        "selection_metric": GTB_TUNING_METRIC,
        "folds": GTB_TUNING_FOLDS,
        "seeds": GTB_TUNING_SEEDS,
        "selected_params": selected_params,
        "best_result": best_row,
    }


# 하나의 validation fold에 대해 sample 추출, GTB 학습, validation 평가, feature importance 계산을 수행한다.
def run_fold(validation_fold):
    samples = get_fold_samples(validation_fold)
    fold_inputs = samples["fold_inputs"]
    train_fc = samples["train_fc"]
    valid_fc = samples["valid_fc"]
    classifier = train_classifier(train_fc, INPUT_BANDS)
    ee_valid_confusion = evaluate_fc(valid_fc, classifier)
    valid_confusion = ee_valid_confusion.getInfo()
    valid_accuracy = ee_valid_confusion.accuracy().getInfo()
    valid_kappa = ee_valid_confusion.kappa().getInfo()
    valid_metrics = sample_metrics(valid_fc, classifier)
    importance, normalized_importance = classifier_importance(
        classifier,
        INPUT_BANDS,
    )
    probability = (
        classify_image_with_voting(
            fold_inputs["feature_image"],
            INPUT_BANDS,
            classifier,
            "flood_prob",
        )
        .clip(seoul)
    )
    hotspot_metrics = compute_hotspot_metrics(
        probability,
        fold_inputs["valid_positive"],
    )
    result = {
        "model": FINAL_MODEL_CONFIG["name"],
        "model_family": FINAL_MODEL_CONFIG["family"],
        "fold": validation_fold,
        "train_positive_count": fold_inputs["train_positive"].size().getInfo(),
        "valid_positive_count": fold_inputs["valid_positive"].size().getInfo(),
        "train_sample_count": train_fc.size().getInfo(),
        "valid_sample_count": valid_fc.size().getInfo(),
        "train_label_histogram": train_fc.aggregate_histogram("label").getInfo(),
        "valid_label_histogram": valid_fc.aggregate_histogram("label").getInfo(),
        "valid_confusion": valid_confusion,
        "valid_accuracy": valid_accuracy,
        "valid_kappa": valid_kappa,
        "valid_metrics": valid_metrics,
        "importance": importance,
        "importance_normalized": normalized_importance,
        "hotspot_metrics": hotspot_metrics,
        "classifier_count": len(as_classifier_list(classifier)),
        "classifier": classifier,
        "probability": probability,
    }
    print(
        {
            "model": result["model"],
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


def run_final_model():
    """Train the final model on all available Seoul flood positives."""
    alpha_score = make_alpha_score(positive_points)
    feature_image = static_features["image"].addBands(alpha_score)
    positive_mask = ee.Image.constant(0).byte().paint(
        positive_points.map(lambda f: f.buffer(POSITIVE_BUFFER_M)),
        1,
    )
    analysis_mask = ee.Image.constant(1).clip(seoul)
    negative_mask = make_negative_mask(NEGATIVE_BUFFER_M)
    train_fc = sample_split(
        feature_image,
        INPUT_BANDS,
        positive_mask,
        negative_mask,
        analysis_mask,
        2700,
    )
    classifier = train_classifier(train_fc, INPUT_BANDS)
    probability = (
        classify_image_with_voting(
            feature_image,
            INPUT_BANDS,
            classifier,
            "flood_prob",
        )
        .clip(seoul)
    )
    importance, normalized_importance = classifier_importance(
        classifier,
        INPUT_BANDS,
    )
    result = {
        "model": FINAL_MODEL_CONFIG["name"],
        "model_family": FINAL_MODEL_CONFIG["family"],
        "train_positive_count": positive_points.size().getInfo(),
        "train_sample_count": train_fc.size().getInfo(),
        "train_label_histogram": train_fc.aggregate_histogram("label").getInfo(),
        "importance": importance,
        "importance_normalized": normalized_importance,
        "classifier_count": len(as_classifier_list(classifier)),
        "classifier": classifier,
        "feature_image": feature_image,
        "probability": probability,
    }
    print(
        {
            "model": result["model"],
            "training_scope": "all_seoul_positive_points",
            "train_sample_count": result["train_sample_count"],
            "classifier_count": result["classifier_count"],
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


def summarize_hotspot_metrics(rows):
    grouped = {}
    for row in rows:
        for metric in row.get("hotspot_metrics", []):
            percentile = int(metric["percentile"])
            group = grouped.setdefault(
                percentile,
                {
                    "percentile": percentile,
                    "top_percent": int(metric["top_percent"]),
                    "point_recalls": [],
                    "all_point_recalls": [],
                    "lifts": [],
                    "area_shares": [],
                    "hit_points": [],
                    "hotspot_areas": [],
                },
            )
            group["point_recalls"].append(metric["evaluated_point_recall"])
            group["all_point_recalls"].append(metric["all_point_recall"])
            group["lifts"].append(metric["evaluated_point_lift"])
            group["area_shares"].append(metric["hotspot_area_share"])
            group["hit_points"].append(metric["hit_points"])
            group["hotspot_areas"].append(metric["hotspot_area_km2"])

    return [
        {
            "percentile": percentile,
            "top_percent": values["top_percent"],
            "evaluated_point_recall_mean": mean(values["point_recalls"]),
            "evaluated_point_recall_std": sample_std(values["point_recalls"]),
            "all_point_recall_mean": mean(values["all_point_recalls"]),
            "evaluated_point_lift_mean": mean(values["lifts"]),
            "hotspot_area_share_mean": mean(values["area_shares"]),
            "hit_points_mean": mean(values["hit_points"]),
            "hotspot_area_km2_mean": mean(values["hotspot_areas"]),
        }
        for percentile, values in sorted(grouped.items())
    ]


# CSV/JSON 저장에 필요한 fold 결과만 남겨 출력 파일을 간결하게 만든다.
def sanitize_fold_result(row):
    return {
        "model": row["model"],
        "model_family": row["model_family"],
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
        "classifier_count": row.get("classifier_count"),
        "hotspot_point_recall": {
            f"top_{int(metric['top_percent'])}_percent": metric["evaluated_point_recall"]
            for metric in row.get("hotspot_metrics", [])
        },
    }


def point_segment_distance(point, start, end):
    px, py = point
    x1, y1 = start
    x2, y2 = end
    dx = x2 - x1
    dy = y2 - y1
    if dx == 0 and dy == 0:
        return ((px - x1) ** 2 + (py - y1) ** 2) ** 0.5

    ratio = ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)
    ratio = max(0, min(1, ratio))
    proj_x = x1 + ratio * dx
    proj_y = y1 + ratio * dy
    return ((px - proj_x) ** 2 + (py - proj_y) ** 2) ** 0.5


def simplify_line(points, tolerance):
    if tolerance <= 0 or len(points) <= 2:
        return points

    keep = [False] * len(points)
    keep[0] = True
    keep[-1] = True
    stack = [(0, len(points) - 1)]
    while stack:
        start_idx, end_idx = stack.pop()
        max_distance = -1
        max_idx = None
        for idx in range(start_idx + 1, end_idx):
            distance = point_segment_distance(
                points[idx],
                points[start_idx],
                points[end_idx],
            )
            if distance > max_distance:
                max_distance = distance
                max_idx = idx
        if max_idx is not None and max_distance > tolerance:
            keep[max_idx] = True
            stack.append((start_idx, max_idx))
            stack.append((max_idx, end_idx))

    return [point for point, should_keep in zip(points, keep) if should_keep]


def simplify_ring(ring, tolerance):
    coords = [(float(point[0]), float(point[1])) for point in ring]
    if len(coords) <= 4 or tolerance <= 0:
        return [[x, y] for x, y in coords]

    if coords[0] == coords[-1]:
        coords = coords[:-1]
    if len(coords) <= 3:
        return [[x, y] for x, y in coords + [coords[0]]]

    xs = [point[0] for point in coords]
    ys = [point[1] for point in coords]
    if max(xs) - min(xs) >= max(ys) - min(ys):
        anchor_a = xs.index(min(xs))
        anchor_b = xs.index(max(xs))
    else:
        anchor_a = ys.index(min(ys))
        anchor_b = ys.index(max(ys))
    if anchor_a > anchor_b:
        anchor_a, anchor_b = anchor_b, anchor_a

    first_path = coords[anchor_a: anchor_b + 1]
    second_path = coords[anchor_b:] + coords[: anchor_a + 1]
    first_half = simplify_line(first_path, tolerance)
    second_half = simplify_line(second_path, tolerance)
    simplified = first_half[:-1] + second_half[:-1]

    deduped = []
    for point in simplified:
        if not deduped or point != deduped[-1]:
            deduped.append(point)
    if len(deduped) < 3:
        start = coords[anchor_a]
        end = coords[anchor_b]
        third = max(
            (point for idx, point in enumerate(coords) if idx not in (anchor_a, anchor_b)),
            key=lambda point: point_segment_distance(point, start, end),
            default=None,
        )
        deduped = [start, third, end] if third else coords[:3]
    if deduped[0] != deduped[-1]:
        deduped.append(deduped[0])
    return [[x, y] for x, y in deduped]


def simplify_geometry(geometry, tolerance):
    if tolerance <= 0:
        return geometry

    geom_type = geometry.get("type")
    if geom_type == "Polygon":
        return {
            **geometry,
            "coordinates": [
                simplify_ring(ring, tolerance)
                for ring in geometry["coordinates"]
            ],
        }
    if geom_type == "MultiPolygon":
        return {
            **geometry,
            "coordinates": [
                [simplify_ring(ring, tolerance) for ring in polygon]
                for polygon in geometry["coordinates"]
            ],
        }
    return geometry


def shape_record_to_ee_feature(shape_record, source_zip, source_shp):
    if shape_record.shape.shapeType == shapefile.NULL:
        return None

    geometry = shape_record.shape.__geo_interface__
    if not geometry or geometry.get("type") == "Null":
        return None
    geometry = simplify_geometry(geometry, OFFICIAL_FLOOD_SHP_SIMPLIFY_M)
    properties = dict(shape_record.record.as_dict())
    properties.update({"source_zip": source_zip, "source_shp": source_shp})
    return ee.Feature(
        ee.Geometry(geometry, OFFICIAL_FLOOD_SHP_PROJ, False),
        properties,
    )


def infer_official_flood_frequency(dataset_name, fallback="unknown"):
    if not dataset_name:
        return fallback
    if "기왕최대" in dataset_name or str(dataset_name).upper() == "MAX":
        return "MAX"
    match = re.search(r"(\d+)\s*년", dataset_name)
    if match:
        return match.group(1)
    match = re.search(r"freq[_-]?(\d+|MAX)", dataset_name, re.IGNORECASE)
    if match:
        value = match.group(1).upper()
        return str(int(value)) if value.isdigit() else value
    return fallback


def official_flood_frequency_sort_key(summary):
    order = {"30": 30, "50": 50, "80": 80, "100": 100, "500": 500, "MAX": 999}
    return order.get(str(summary.get("flood_frequency", "unknown")).upper(), 10000)


def official_flood_zip_matches_frequency(zip_path, frequency):
    frequency = str(frequency or "unknown").upper()
    if frequency == "UNKNOWN":
        return True
    suffix_by_frequency = {
        "30": "_030.zip",
        "50": "_050.zip",
        "80": "_080.zip",
        "100": "_100.zip",
        "500": "_500.zip",
        "MAX": "_MAX.zip",
    }
    suffix = suffix_by_frequency.get(frequency)
    return True if not suffix else os.path.basename(zip_path).upper().endswith(suffix.upper())


def official_flood_dir_metadata(zip_dir):
    metadata_path = os.path.join(zip_dir, "metadata.json")
    metadata = {}
    if os.path.exists(metadata_path):
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

    dataset = metadata.get("dataset") or os.path.basename(zip_dir)
    frequency = metadata.get("frequency") or infer_official_flood_frequency(
        dataset,
        fallback=infer_official_flood_frequency(os.path.basename(zip_dir)),
    )
    label = metadata.get("label") or (
        "기왕최대" if frequency == "MAX" else f"{frequency}년"
    )
    return {
        "official_dataset": dataset,
        "flood_frequency": frequency,
        "flood_frequency_label": label,
    }


def discover_official_flood_shp_dirs():
    dirs = []
    if OFFICIAL_FLOOD_SHP_FREQ_ROOT_DIR and os.path.isdir(OFFICIAL_FLOOD_SHP_FREQ_ROOT_DIR):
        for name in sorted(os.listdir(OFFICIAL_FLOOD_SHP_FREQ_ROOT_DIR)):
            candidate = os.path.join(OFFICIAL_FLOOD_SHP_FREQ_ROOT_DIR, name)
            if not os.path.isdir(candidate):
                continue
            if any(item.lower().endswith(".zip") for item in os.listdir(candidate)):
                dirs.append(candidate)

    if not dirs and OFFICIAL_FLOOD_SHP_ZIP_DIR:
        dirs.append(OFFICIAL_FLOOD_SHP_ZIP_DIR)

    return sorted(
        dirs,
        key=lambda path: official_flood_frequency_sort_key(
            official_flood_dir_metadata(path)
        ),
    )


def load_shp_zip_feature_collection(zip_dir):
    metadata = official_flood_dir_metadata(zip_dir)
    if not os.path.isdir(zip_dir):
        return None, {
            "used": False,
            "source_type": "shp_zip_dir",
            "source": zip_dir,
            "zip_count": 0,
            "feature_count": 0,
            "reason": "directory_missing",
        }

    zip_paths = sorted(
        os.path.join(zip_dir, name)
        for name in os.listdir(zip_dir)
        if name.lower().endswith(".zip")
    )
    zip_paths = [
        path for path in zip_paths
        if official_flood_zip_matches_frequency(path, metadata.get("flood_frequency"))
    ]
    if not zip_paths:
        return None, {
            "used": False,
            "source_type": "shp_zip_dir",
            "source": zip_dir,
            **metadata,
            "zip_count": 0,
            "feature_count": 0,
            "reason": "zip_files_missing",
        }

    feature_collections = []
    chunk_summaries = []
    shp_file_count = 0
    total_count = 0
    for zip_path in zip_paths:
        zip_features = []
        with zipfile.ZipFile(zip_path) as zf, tempfile.TemporaryDirectory() as temp_dir:
            zf.extractall(temp_dir)
            shp_paths = []
            for root, _, files in os.walk(temp_dir):
                shp_paths.extend(
                    os.path.join(root, name)
                    for name in files
                    if name.lower().endswith(".shp")
                )
            shp_file_count += len(shp_paths)

            for shp_path in shp_paths:
                reader = shapefile.Reader(
                    shp_path,
                    encoding=OFFICIAL_FLOOD_SHP_ENCODING,
                )
                for shape_record in reader.iterShapeRecords():
                    feature = shape_record_to_ee_feature(
                        shape_record,
                        os.path.basename(zip_path),
                        os.path.basename(shp_path),
                    )
                    if feature is not None:
                        zip_features.append(feature)

        if not zip_features:
            continue
        chunk_fc = ee.FeatureCollection(zip_features)
        feature_collections.append(chunk_fc)
        total_count += len(zip_features)
        chunk_summaries.append(
            {
                "source_zip": os.path.basename(zip_path),
                "feature_count": len(zip_features),
            }
        )

    return feature_collections, {
        "used": total_count > 0,
        "source_type": "shp_zip_dir",
        "source": zip_dir,
        **metadata,
        "projection": OFFICIAL_FLOOD_SHP_PROJ,
        "encoding": OFFICIAL_FLOOD_SHP_ENCODING,
        "zip_count": len(zip_paths),
        "shp_file_count": shp_file_count,
        "feature_count": total_count,
        "chunk_count": len(feature_collections),
        "chunks": chunk_summaries,
        "validation_mode": "chunked_zip_sum",
        "reason": None if total_count > 0 else "empty_after_filter_bounds",
    }


def load_external_flood_references():
    references = []
    for zip_dir in discover_official_flood_shp_dirs():
        external_fc, summary = load_shp_zip_feature_collection(zip_dir)
        references.append((external_fc, summary))

    if references:
        return references

    return [
        (
            None,
            {
                "used": False,
                "source_type": None,
                "source": None,
                "feature_count": 0,
                "reason": "not_configured",
            },
        )
    ]


def normalize_external_feature_collections(external_fc):
    if external_fc is None:
        return []
    if isinstance(external_fc, (list, tuple)):
        return [item for item in external_fc if item is not None]
    return [external_fc]


def area_sum_km2(mask):
    value = (
        ee.Image.pixelArea()
        .divide(1e6)
        .updateMask(mask)
        .reduceRegion(
            reducer=ee.Reducer.sum(),
            geometry=seoul,
            scale=ANALYSIS_SCALE,
            maxPixels=1e9,
            bestEffort=True,
            tileScale=4,
        )
        .get("area")
    )
    return ee.Number(ee.Algorithms.If(value, value, 0))


def compute_external_flood_validation(probability, external_fc, reference_summary=None):
    reference_summary = reference_summary or {}
    external_fcs = normalize_external_feature_collections(external_fc)
    probability_mask = probability.mask()
    analysis_mask = ee.Image.constant(1).clip(seoul).updateMask(probability_mask)
    pixel_area_km2 = ee.Image.pixelArea().divide(1e6)
    analysis_summary = (
        pixel_area_km2
        .rename("analysis_area")
        .updateMask(analysis_mask)
        .addBands(
            probability
            .multiply(pixel_area_km2)
            .rename("total_probability_area")
            .updateMask(analysis_mask)
        )
        .reduceRegion(
            reducer=ee.Reducer.sum(),
            geometry=seoul,
            scale=ANALYSIS_SCALE,
            maxPixels=1e9,
            bestEffort=True,
            tileScale=4,
        )
        .getInfo()
    )
    analysis_area = analysis_summary.get("analysis_area", 0) or 0
    total_probability_area = analysis_summary.get("total_probability_area", 0) or 0
    percentile_values = probability.reduceRegion(
        reducer=ee.Reducer.percentile(EXTERNAL_VALIDATION_PERCENTILES),
        geometry=seoul,
        scale=ANALYSIS_SCALE,
        maxPixels=1e9,
        bestEffort=True,
        tileScale=4,
    ).getInfo()
    thresholds = {
        percentile: percentile_values.get(f"flood_prob_p{percentile}")
        for percentile in EXTERNAL_VALIDATION_PERCENTILES
    }

    hotspot_masks = {}
    hotspot_area_bands = []
    for percentile in EXTERNAL_VALIDATION_PERCENTILES:
        threshold = thresholds[percentile]
        hotspot_mask = probability.gte(threshold).selfMask()
        hotspot_masks[percentile] = hotspot_mask
        hotspot_area_bands.append(
            pixel_area_km2
            .rename(f"hotspot_area_{percentile}")
            .updateMask(hotspot_mask)
        )
    hotspot_area_info = (
        ee.Image.cat(hotspot_area_bands)
        .reduceRegion(
            reducer=ee.Reducer.sum(),
            geometry=seoul,
            scale=ANALYSIS_SCALE,
            maxPixels=1e9,
            bestEffort=True,
            tileScale=4,
        )
        .getInfo()
    )
    hotspot_areas = {
        percentile: hotspot_area_info.get(f"hotspot_area_{percentile}", 0) or 0
        for percentile in EXTERNAL_VALIDATION_PERCENTILES
    }

    external_area = 0
    inside_probability_area = 0
    overlap_areas = {percentile: 0 for percentile in EXTERNAL_VALIDATION_PERCENTILES}
    for chunk_fc in external_fcs:
        external_mask = (
            ee.Image.constant(0)
            .byte()
            .paint(chunk_fc, 1)
            .rename("external_flood")
            .clip(seoul)
            .updateMask(probability_mask)
        )
        external_self_mask = external_mask.unmask(0).eq(1).selfMask()
        chunk_bands = [
            pixel_area_km2
            .rename("external_area")
            .updateMask(external_self_mask),
            probability
            .multiply(pixel_area_km2)
            .rename("inside_probability_area")
            .updateMask(external_self_mask),
        ]
        for percentile, hotspot_mask in hotspot_masks.items():
            chunk_bands.append(
                pixel_area_km2
                .rename(f"overlap_area_{percentile}")
                .updateMask(hotspot_mask)
                .updateMask(external_self_mask)
            )
        chunk_summary = (
            ee.Image.cat(chunk_bands)
            .reduceRegion(
                reducer=ee.Reducer.sum(),
                geometry=seoul,
                scale=ANALYSIS_SCALE,
                maxPixels=1e9,
                bestEffort=True,
                tileScale=4,
            )
            .getInfo()
        )
        chunk_area = chunk_summary.get("external_area", 0) or 0
        if not chunk_area:
            continue
        external_area += chunk_area
        inside_probability_area += chunk_summary.get("inside_probability_area", 0) or 0
        for percentile in EXTERNAL_VALIDATION_PERCENTILES:
            overlap_areas[percentile] += (
                chunk_summary.get(f"overlap_area_{percentile}", 0) or 0
            )

    outside_area = max(analysis_area - external_area, 0)
    outside_probability_area = max(total_probability_area - inside_probability_area, 0)
    external_area_share = external_area / analysis_area if analysis_area else None
    inside_mean = inside_probability_area / external_area if external_area else None
    outside_mean = outside_probability_area / outside_area if outside_area else None
    mean_probability_delta = (
        inside_mean - outside_mean
        if inside_mean is not None and outside_mean is not None
        else None
    )

    rows = []
    for percentile in EXTERNAL_VALIDATION_PERCENTILES:
        hotspot_area = hotspot_areas[percentile]
        overlap_area = overlap_areas[percentile]
        hotspot_precision = overlap_area / hotspot_area if hotspot_area else None
        external_recall = overlap_area / external_area if external_area else None
        hotspot_area_share = hotspot_area / analysis_area if analysis_area else None
        lift = (
            hotspot_precision / external_area_share
            if hotspot_precision is not None and external_area_share
            else None
        )
        rows.append(
            {
                "flood_frequency": reference_summary.get("flood_frequency"),
                "flood_frequency_label": reference_summary.get("flood_frequency_label"),
                "official_dataset": reference_summary.get("official_dataset"),
                "reference_source_type": reference_summary.get("source_type"),
                "reference_source": reference_summary.get("source"),
                "external_validation_mode": reference_summary.get("validation_mode"),
                "external_validation_chunk_count": len(external_fcs),
                "percentile": percentile,
                "top_percent": 100 - percentile,
                "threshold": thresholds[percentile],
                "analysis_area_km2": analysis_area,
                "external_flood_area_km2": external_area,
                "external_flood_area_share": external_area_share,
                "hotspot_area_km2": hotspot_area,
                "hotspot_area_share": hotspot_area_share,
                "overlap_area_km2": overlap_area,
                "hotspot_precision_vs_external": hotspot_precision,
                "external_flood_recall": external_recall,
                "overlap_lift": lift,
                "mean_probability_inside_external": inside_mean,
                "mean_probability_outside_external": outside_mean,
                "mean_probability_delta": mean_probability_delta,
            }
        )

    return rows


def run_external_validation(probability):
    if not RUN_EXTERNAL_VALIDATION:
        return {
            "used": False,
            "reason": "not_requested",
        }, []

    external_references = load_external_flood_references()
    summaries = [summary for _, summary in external_references]
    reference_summary = {
        "used": any(summary.get("used") for summary in summaries),
        "reference_count": len(summaries),
        "references": summaries,
        "reason": None if any(summary.get("used") for summary in summaries) else "not_configured",
    }
    rows = []
    if not reference_summary["used"]:
        return reference_summary, rows

    for external_fc, summary in external_references:
        if not summary.get("used"):
            continue
        print(
            "External validation reference:",
            {
                "frequency": summary.get("flood_frequency_label"),
                "feature_count": summary.get("feature_count"),
            },
        )
        rows.extend(compute_external_flood_validation(probability, external_fc, summary))
    return reference_summary, rows


def run_final_map_outputs(analysis_result):
    try:
        from . import create_final_flood_risk_map
    except ImportError:
        import create_final_flood_risk_map

    return create_final_flood_risk_map.create_map_outputs(
        analysis_result,
        sys.modules[__name__],
    )


# 전체 실행 진입점: 5-fold 공간 검증을 돌리고 metric/importance/output 경로를 저장한다.
def main(generate_final_map=None):
    if generate_final_map is None:
        generate_final_map = os.environ.get("GENERATE_FINAL_MAP", "1") == "1"

    print("Analysis model bands:", INPUT_BANDS)
    print("Final model:", FINAL_MODEL_CONFIG["name"])
    print("Positive fold histogram:", positive_data["fold_histogram"])
    print("AlphaEarth tiles:", alpha_tile_count)

    tuning_rows = []
    tuning_summary = {
        "used": False,
        "reason": "not_requested",
    }
    if RUN_HYPERPARAMETER_TUNING:
        tuning_rows, tuning_summary = run_hyperparameter_tuning()
    print("Final model parameters:", FINAL_MODEL_CONFIG["params"])

    cv_rows = [run_fold(fold) for fold in range(SPATIAL_FOLDS)]
    cv_summary = summarize_cv(cv_rows)
    importance_summary = summarize_importance(cv_rows)
    topk_summary = summarize_hotspot_metrics(cv_rows)
    cv_result_rows = [sanitize_fold_result(row) for row in cv_rows]
    final_model = run_final_model()
    external_reference_summary, external_validation_rows = run_external_validation(
        final_model["probability"]
    )

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
            "hotspot_eval_percentiles": HOTSPOT_EVAL_PERCENTILES,
            "run_hyperparameter_tuning": RUN_HYPERPARAMETER_TUNING,
            "gtb_tuning_metric": GTB_TUNING_METRIC,
            "gtb_tuning_folds": GTB_TUNING_FOLDS,
            "gtb_tuning_seeds": GTB_TUNING_SEEDS,
            "gtb_optuna_trials": GTB_OPTUNA_TRIALS,
            "gtb_optuna_random_seed": GTB_OPTUNA_RANDOM_SEED,
            "gtb_optuna_startup_trials": GTB_OPTUNA_STARTUP_TRIALS,
            "gtb_optuna_pruning": GTB_OPTUNA_PRUNING,
            "gtb_optuna_warmup_folds": GTB_OPTUNA_WARMUP_FOLDS,
            "gtb_optuna_trees_range": [
                GTB_OPTUNA_TREES_MIN,
                GTB_OPTUNA_TREES_MAX,
                GTB_OPTUNA_TREES_STEP,
            ],
            "gtb_optuna_shrinkage_range": [
                GTB_OPTUNA_SHRINKAGE_MIN,
                GTB_OPTUNA_SHRINKAGE_MAX,
            ],
            "gtb_optuna_sampling_rate_range": [
                GTB_OPTUNA_SAMPLING_RATE_MIN,
                GTB_OPTUNA_SAMPLING_RATE_MAX,
            ],
            "gtb_optuna_max_nodes_choices": GTB_OPTUNA_MAX_NODES_CHOICES,
            "reference_geojson": REFERENCE_GEOJSON,
            "pump_station_csv": PUMP_STATION_CSV,
            "sewer_sensor_gu_stats_csv": SEWER_SENSOR_GU_STATS_CSV,
            "sewer_level_sensor_csv": SEWER_LEVEL_SENSOR_CSV,
            "run_external_validation": RUN_EXTERNAL_VALIDATION,
            "generate_final_map": generate_final_map,
            "official_flood_shp_freq_root_dir": OFFICIAL_FLOOD_SHP_FREQ_ROOT_DIR,
            "official_flood_shp_zip_dir": OFFICIAL_FLOOD_SHP_ZIP_DIR,
            "official_flood_shp_proj": OFFICIAL_FLOOD_SHP_PROJ,
            "official_flood_shp_simplify_m": OFFICIAL_FLOOD_SHP_SIMPLIFY_M,
            "external_validation_percentiles": EXTERNAL_VALIDATION_PERCENTILES,
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
            "classifier": FINAL_MODEL_CONFIG,
            "bands": INPUT_BANDS,
            "final_training": {
                "scope": "all_seoul_positive_points",
                "train_positive_count": final_model["train_positive_count"],
                "train_sample_count": final_model["train_sample_count"],
                "train_label_histogram": final_model["train_label_histogram"],
                "classifier_count": final_model["classifier_count"],
            },
        },
        "validation": {
            "summary": cv_summary,
            "fold_results": cv_result_rows,
        },
        "feature_importance": importance_summary,
        "topk": {
            "cv_summary": topk_summary,
        },
        "hyperparameter_tuning": {
            **tuning_summary,
            "results": tuning_rows,
        },
        "external_validation": {
            "reference": external_reference_summary,
            "overlap_metrics": external_validation_rows,
        },
        "outputs": {
            "output_dir": OUTPUT_DIR,
            "metrics_json": METRICS_JSON,
            "cv_results_csv": CV_RESULTS_CSV,
            "feature_importance_csv": FEATURE_IMPORTANCE_CSV,
            "topk_summary_csv": TOPK_SUMMARY_CSV,
            "external_validation_csv": EXTERNAL_VALIDATION_CSV,
            "hyperparameter_tuning_csv": HYPERPARAMETER_TUNING_CSV,
        },
    }

    analysis_result = {
        "metrics_payload": metrics_payload,
        "cv_rows": cv_rows,
        "cv_summary": cv_summary,
        "importance_summary": importance_summary,
        "topk_summary": topk_summary,
        "final_model": final_model,
        "external_reference_summary": external_reference_summary,
        "external_validation_rows": external_validation_rows,
        "final_map_outputs": {},
    }
    if generate_final_map:
        final_map_outputs = run_final_map_outputs(analysis_result)
        metrics_payload["outputs"].update(final_map_outputs)
        analysis_result["final_map_outputs"] = final_map_outputs
    else:
        print("Final map generation skipped: GENERATE_FINAL_MAP=0")

    write_json(METRICS_JSON, metrics_payload)
    write_csv(CV_RESULTS_CSV, cv_result_rows)
    write_csv(FEATURE_IMPORTANCE_CSV, importance_summary)
    write_csv(TOPK_SUMMARY_CSV, topk_summary)
    write_csv(EXTERNAL_VALIDATION_CSV, external_validation_rows)
    write_csv(HYPERPARAMETER_TUNING_CSV, tuning_rows)
    print("Validation summary:", cv_summary)
    print("Top-k summary:", topk_summary)
    if RUN_EXTERNAL_VALIDATION:
        print("External validation references:", external_reference_summary)
    print("Saved:", metrics_payload["outputs"])
    return analysis_result


if __name__ == "__main__":
    main()
