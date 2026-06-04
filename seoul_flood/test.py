import csv
import json
import os
import html
import re
import tempfile
import zipfile

import ee
import geemap
import shapefile


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


def parse_str_list(raw_value):
    """쉼표로 구분된 문자열 환경변수 값을 리스트로 변환한다."""
    return [
        value.strip()
        for value in raw_value.split(",")
        if value.strip()
    ]


def json_safe(value):
    """JSON 저장 가능한 기본 타입만 남기도록 값들을 정리한다."""
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def write_json(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(json_safe(payload), f, ensure_ascii=False, indent=2)


def write_csv(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not rows:
        with open(path, "w", encoding="utf-8", newline="") as f:
            f.write("")
        return

    fieldnames = []
    for row in rows:
        for key in row.keys():
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


def parse_number(value):
    """공개데이터 CSV의 숫자 문자열을 float으로 변환한다."""
    cleaned = str(value or "").replace(",", "").strip()
    if not cleaned:
        return 0.0
    try:
        return float(cleaned)
    except ValueError:
        return 0.0


def normalize_seoul_gu_name(value):
    """서울 자치구 이름만 표준화해서 반환한다."""
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
    if matches:
        return sorted(matches)[0][1]
    return None


def normalize_gu_stats(raw_stats_by_gu, metric_names):
    """구별 원자료를 0~1 feature로 정규화한다."""
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
            band_name = f"{metric}_norm"
            normalized[gu_name][band_name] = (
                raw_stats.get(metric, 0.0) / max_value if max_value else 0.0
            )
    return normalized, max_by_metric


def read_csv_with_fallback(path, encodings=("utf-8-sig", "cp949")):
    last_error = None
    for encoding in encodings:
        try:
            with open(path, "r", encoding=encoding, newline="") as f:
                return list(csv.DictReader(f)), encoding
        except UnicodeDecodeError as error:
            last_error = error
    raise last_error


def load_pump_station_gu_stats(csv_path, min_districts):
    """배수펌프장 CSV를 자치구별 정적 feature 후보로 요약한다."""
    summary = {
        "dataset": "pump_station_gu_stats",
        "path": csv_path,
        "exists": os.path.exists(csv_path),
        "raw_records": 0,
        "unique_stations": 0,
        "district_count": 0,
        "used": False,
        "reason": None,
    }
    if not summary["exists"]:
        summary["reason"] = "file_missing"
        return {}, summary

    rows, encoding = read_csv_with_fallback(csv_path, encodings=("cp949", "utf-8-sig"))
    summary["encoding"] = encoding
    summary["raw_records"] = len(rows)
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
        facility_name = (row.get("시설물명") or "").strip()
        key = (gu_name, facility_name, address)
        station_stats = unique_stations.setdefault(
            key,
            {
                "pump_station_count": 1.0,
                "pump_capacity_sum": 0.0,
                "pump_catchment_area_sum": 0.0,
                "pump_reservoir_capacity_sum": 0.0,
            },
        )
        # 같은 펌프장이 배수문별 행으로 반복되는 경우가 많아, 중복 행은 최대값만 보존한다.
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
    normalized_stats_by_gu, max_by_metric = normalize_gu_stats(raw_stats_by_gu, metric_names)
    summary.update(
        {
            "unique_stations": len(unique_stations),
            "district_count": len(raw_stats_by_gu),
            "skipped_rows": skipped_rows,
            "raw_stats_by_gu": raw_stats_by_gu,
            "max_by_metric": max_by_metric,
        }
    )
    if summary["district_count"] < min_districts:
        summary["reason"] = "too_few_districts"
        return {}, summary

    summary["used"] = True
    return normalized_stats_by_gu, summary


def load_sewer_sensor_gu_stats(summary_csv_path, raw_csv_path, min_districts):
    """하수관로 수위 자료에서 자치구별 관측소 수 feature 후보를 만든다."""
    summary = {
        "dataset": "sewer_level_sensor_gu_stats",
        "summary_path": summary_csv_path,
        "raw_path": raw_csv_path,
        "summary_exists": os.path.exists(summary_csv_path),
        "raw_exists": os.path.exists(raw_csv_path),
        "district_count": 0,
        "used": False,
        "reason": None,
    }

    raw_stats_by_gu = {}
    if summary["summary_exists"]:
        rows, encoding = read_csv_with_fallback(summary_csv_path)
        summary["encoding"] = encoding
        summary["source"] = "summary_csv"
        for row in rows:
            gu_name = normalize_seoul_gu_name(row.get("gu") or row.get("구분명"))
            if gu_name is None:
                continue
            raw_stats_by_gu[gu_name] = {
                "sewer_sensor_count": parse_number(row.get("sensor_count")),
            }
    elif summary["raw_exists"]:
        summary["source"] = "raw_csv"
        sensor_ids_by_gu = {}
        last_error = None
        for encoding in ("cp949", "utf-8-sig"):
            try:
                with open(raw_csv_path, "r", encoding=encoding, newline="") as f:
                    reader = csv.DictReader(f)
                    id_field = reader.fieldnames[0] if reader.fieldnames else None
                    if id_field is None:
                        continue
                    for row in reader:
                        gu_name = normalize_seoul_gu_name(row.get("구분명"))
                        sensor_id = str(row.get(id_field) or "").strip()
                        if gu_name and sensor_id:
                            sensor_ids_by_gu.setdefault(gu_name, set()).add(sensor_id)
                summary["encoding"] = encoding
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

    normalized_stats_by_gu, max_by_metric = normalize_gu_stats(
        raw_stats_by_gu,
        ["sewer_sensor_count"],
    )
    summary.update(
        {
            "district_count": len(raw_stats_by_gu),
            "raw_stats_by_gu": raw_stats_by_gu,
            "max_by_metric": max_by_metric,
        }
    )
    if summary["district_count"] < min_districts:
        summary["reason"] = "too_few_districts"
        return {}, summary

    summary["used"] = True
    return normalized_stats_by_gu, summary


# 실행 환경에서 조정할 수 있는 주요 설정값들이다.
# EE_PROJECT_ID만 필수이고, 나머지는 없으면 기본값을 사용한다.
PROJECT_ID = os.environ.get("EE_PROJECT_ID")
if not PROJECT_ID:
    raise ValueError("EE_PROJECT_ID is missing. Set it in .env or your shell environment.")

YEAR = int(os.environ.get("YEAR", "2024"))
OUTPUT_HTML = resolve_output_path(os.environ.get("OUTPUT_HTML", "seoul_flood_risk_rf.html"))
OUTPUT_DIR = resolve_output_path(os.environ.get("OUTPUT_DIR", "outputs"))
METRICS_JSON = os.path.join(OUTPUT_DIR, "metrics.json")
CV_RESULTS_CSV = os.path.join(OUTPUT_DIR, "cv_results.csv")
FEATURE_IMPORTANCE_CSV = os.path.join(OUTPUT_DIR, "feature_importance.csv")
TOPK_SUMMARY_CSV = os.path.join(OUTPUT_DIR, "topk_summary.csv")
SELECTED_FOLD_TOPK_CSV = os.path.join(OUTPUT_DIR, "selected_fold_topk.csv")
RISK_GRADE_SUMMARY_CSV = os.path.join(OUTPUT_DIR, "risk_grade_summary.csv")
RISK_GRADE_POINTS_CSV = os.path.join(OUTPUT_DIR, "risk_grade_points.csv")
CUMULATIVE_RISK_GRADE_POINTS_CSV = os.path.join(
    OUTPUT_DIR,
    "cumulative_risk_grade_points.csv",
)
ANALYSIS_SCALE = int(os.environ.get("ANALYSIS_SCALE", "30"))
BOUNDARY_BUFFER_M = int(os.environ.get("BOUNDARY_BUFFER_M", "0"))
if BOUNDARY_BUFFER_M < 0:
    raise ValueError("BOUNDARY_BUFFER_M는 0 이상의 정수여야 합니다.")
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
HOTSPOT_EVAL_PERCENTILES = parse_int_list(
    os.environ.get("HOTSPOT_EVAL_PERCENTILES", "80,90,95")
)
RISK_GRADE_PERCENTILES = parse_int_list(
    os.environ.get("RISK_GRADE_PERCENTILES", "50,75,90,95")
)
if (
    RISK_GRADE_PERCENTILES != sorted(RISK_GRADE_PERCENTILES)
    or not all(0 < percentile < 100 for percentile in RISK_GRADE_PERCENTILES)
):
    raise ValueError("RISK_GRADE_PERCENTILES는 0과 100 사이의 오름차순 정수여야 합니다.")
RISK_GRADE_PALETTE = parse_str_list(
    os.environ.get("RISK_GRADE_PALETTE", "#2b83ba,#abdda4,#ffffbf,#fdae61,#d7191c")
)
RISK_GRADE_NAMES = parse_str_list(
    os.environ.get("RISK_GRADE_NAMES", "Very low,Low,Moderate,High,Very high")
)
RISK_GRADE_COUNT = len(RISK_GRADE_PERCENTILES) + 1
if len(RISK_GRADE_PALETTE) != RISK_GRADE_COUNT:
    raise ValueError("RISK_GRADE_PALETTE 색상 수는 위험도 등급 수와 같아야 합니다.")
if len(RISK_GRADE_NAMES) != RISK_GRADE_COUNT:
    raise ValueError("RISK_GRADE_NAMES 이름 수는 위험도 등급 수와 같아야 합니다.")
SPATIAL_BLOCK_DEGREES = float(os.environ.get("SPATIAL_BLOCK_DEGREES", "0.015"))
SPATIAL_FOLDS = int(os.environ.get("SPATIAL_FOLDS", "5"))
VALIDATION_FOLD = int(os.environ.get("VALIDATION_FOLD", "0"))
RUN_FULL_CV = os.environ.get("RUN_FULL_CV", "0") == "1"
EVALUATION_FOLDS = parse_int_list(
    os.environ.get(
        "EVALUATION_FOLDS",
        (
            ",".join(str(fold) for fold in range(SPATIAL_FOLDS))
            if RUN_FULL_CV
            else str(VALIDATION_FOLD)
        ),
    )
)
EVALUATION_FOLDS = [
    fold for fold in EVALUATION_FOLDS if 0 <= fold < SPATIAL_FOLDS
]
if not EVALUATION_FOLDS:
    raise ValueError("EVALUATION_FOLDS에 유효한 fold가 없습니다.")
RUN_FEATURE_ABLATIONS = os.environ.get(
    "RUN_FEATURE_ABLATIONS",
    "1" if RUN_FULL_CV else "0",
) == "1"
RUN_WATER_FEATURE_ABLATIONS = os.environ.get(
    "RUN_WATER_FEATURE_ABLATIONS",
    "1" if RUN_FEATURE_ABLATIONS else "0",
) == "1"
RUN_DRAINAGE_ABLATION = os.environ.get(
    "RUN_DRAINAGE_ABLATION",
    "1" if RUN_FEATURE_ABLATIONS else "0",
) == "1"
RUN_ALPHA_ABLATION = os.environ.get("RUN_ALPHA_ABLATION", "0") == "1"
RUN_HOTSPOT_EVAL = os.environ.get("RUN_HOTSPOT_EVAL", "1") == "1"
HOTSPOT_EVAL_IN_CV = os.environ.get(
    "HOTSPOT_EVAL_IN_CV",
    "1" if RUN_FULL_CV else "0",
) == "1"
GENERATE_MAP_OUTPUTS = os.environ.get("GENERATE_MAP_OUTPUTS", "1") == "1"
RUN_COVERAGE_DIAGNOSTICS = os.environ.get("RUN_COVERAGE_DIAGNOSTICS", "0") == "1"
RUN_BUFFER_SENSITIVITY = os.environ.get("RUN_BUFFER_SENSITIVITY", "0") == "1"
POSITIVE_BUFFER_SWEEP_M = parse_int_list(
    os.environ.get("POSITIVE_BUFFER_SWEEP_M", "30,60,90")
)
NEGATIVE_BUFFER_SWEEP_M = parse_int_list(
    os.environ.get("NEGATIVE_BUFFER_SWEEP_M", "200,300,500")
)
BUFFER_SENSITIVITY_FOLDS = parse_int_list(
    os.environ.get("BUFFER_SENSITIVITY_FOLDS", str(VALIDATION_FOLD))
)
WATER_DISTANCE_PIXELS = int(os.environ.get("WATER_DISTANCE_PIXELS", "256"))
DRAINAGE_FEATURE_MODE = os.environ.get("DRAINAGE_FEATURE_MODE", "gu_stats").strip().lower()
if DRAINAGE_FEATURE_MODE not in {"none", "points", "gu_stats", "all"}:
    raise ValueError("DRAINAGE_FEATURE_MODE는 none, points, gu_stats, all 중 하나여야 합니다.")
DRAINAGE_INFRA_ENABLED = os.environ.get(
    "DRAINAGE_INFRA_ENABLED",
    "1" if RUN_DRAINAGE_ABLATION or DRAINAGE_FEATURE_MODE in {"points", "all"} else "0",
) == "1"
DRAINAGE_INFRA_RADIUS_M = int(os.environ.get("DRAINAGE_INFRA_RADIUS_M", "500"))
DRAINAGE_INFRA_MIN_ACTIVE_POINTS = int(
    os.environ.get("DRAINAGE_INFRA_MIN_ACTIVE_POINTS", "500")
)
DRAINAGE_GU_STATS_ENABLED = os.environ.get(
    "DRAINAGE_GU_STATS_ENABLED",
    "1" if RUN_DRAINAGE_ABLATION or DRAINAGE_FEATURE_MODE in {"gu_stats", "all"} else "0",
) == "1"
DRAINAGE_GU_STATS_MIN_DISTRICTS = int(
    os.environ.get("DRAINAGE_GU_STATS_MIN_DISTRICTS", "5")
)
LID_PRECONSULT_ZIP = resolve_input_path(
    os.environ.get("LID_PRECONSULT_ZIP", "source/SP_BNLC_AS.zip"),
    fallback_dir=os.path.join(SCRIPT_DIR, "source"),
)
RAINWATER_USE_ZIP = resolve_input_path(
    os.environ.get("RAINWATER_USE_ZIP", "source/SP_LGRC_AS.zip"),
    fallback_dir=os.path.join(SCRIPT_DIR, "source"),
)
PUMP_STATION_CSV = resolve_input_path(
    os.environ.get("PUMP_STATION_CSV", "source/seoul_pump_stations.csv"),
    fallback_dir=os.path.join(SCRIPT_DIR, "source"),
)
SEWER_SENSOR_GU_STATS_CSV = resolve_input_path(
    os.environ.get("SEWER_SENSOR_GU_STATS_CSV", "source/sewer_level_sensor_gu_stats.csv"),
    fallback_dir=os.path.join(SCRIPT_DIR, "source"),
)
SEWER_LEVEL_SENSOR_CSV = resolve_input_path(
    os.environ.get("SEWER_LEVEL_SENSOR_CSV", "source/sewer_level_202605.csv"),
    fallback_dir=os.path.join(SCRIPT_DIR, "source"),
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


def load_epsg5186_point_zip(zip_path, dataset_name, min_active_points):
    """서울시 EPSG:5186 shapefile zip을 EE point FeatureCollection으로 변환한다."""
    summary = {
        "dataset": dataset_name,
        "path": zip_path,
        "exists": os.path.exists(zip_path),
        "raw_records": 0,
        "active_points": 0,
        "used": False,
        "reason": None,
    }
    if not summary["exists"]:
        summary["reason"] = "file_missing"
        return ee.FeatureCollection([]), summary

    features = []
    with tempfile.TemporaryDirectory() as tmp_dir:
        with zipfile.ZipFile(zip_path) as zip_file:
            zip_file.extractall(tmp_dir)

        shp_paths = [
            os.path.join(tmp_dir, name)
            for name in os.listdir(tmp_dir)
            if name.lower().endswith(".shp")
        ]
        if not shp_paths:
            summary["reason"] = "shapefile_missing"
            return ee.FeatureCollection([]), summary

        reader = shapefile.Reader(shp_paths[0], encoding="cp949")
        fields = [field[0] for field in reader.fields[1:]]
        del_idx = fields.index("DEL_YN") if "DEL_YN" in fields else None
        summary["raw_records"] = len(reader)

        for shape_record in reader.iterShapeRecords():
            if del_idx is not None and str(shape_record.record[del_idx]).upper() == "Y":
                continue
            for point in shape_record.shape.points:
                features.append(
                    ee.Feature(
                        ee.Geometry.Point(
                            point,
                            ee.Projection("EPSG:5186"),
                        ),
                        {"one": 1, "dataset": dataset_name},
                    )
                )

    summary["active_points"] = len(features)
    if summary["active_points"] < min_active_points:
        summary["reason"] = "too_few_active_points"
        return ee.FeatureCollection([]), summary

    summary["used"] = True
    return ee.FeatureCollection(features), summary


def make_point_density_feature(point_fc, band_name, radius_m):
    """주변 반경 내 point 개수를 0~1 범위의 밀도 feature로 만든다."""
    raw_band = f"{band_name}_raw"
    point_image = (
        point_fc.reduceToImage(["one"], ee.Reducer.sum())
        .unmask(0)
        .rename(f"{band_name}_points")
        .clip(seoul)
        .reproject(crs=dem.projection(), scale=ANALYSIS_SCALE)
    )
    density_raw = (
        point_image.reduceNeighborhood(
            reducer=ee.Reducer.sum(),
            kernel=ee.Kernel.circle(radius=radius_m, units="meters"),
        )
        .rename(raw_band)
        .clip(seoul)
    )
    stats = density_raw.reduceRegion(
        reducer=ee.Reducer.percentile([95]),
        geometry=seoul,
        scale=ANALYSIS_SCALE,
        maxPixels=1e9,
        bestEffort=True,
        tileScale=4,
    ).getInfo()
    p95 = stats.get(f"{raw_band}_p95") or 1
    p95 = max(float(p95), 1)
    density = density_raw.divide(p95).clamp(0, 1).rename(band_name)
    return density, p95


def make_gu_stat_feature_images(stats_by_gu, dataset_name):
    """자치구별 통계 feature를 ADM2 경계에 칠한 raster band로 변환한다."""
    band_names = sorted(
        {
            band_name
            for gu_stats in stats_by_gu.values()
            for band_name in gu_stats
        }
    )
    summary = {
        "dataset": dataset_name,
        "used": False,
        "district_count": len(stats_by_gu),
        "band_names": band_names,
        "matched_adm2_count": 0,
        "reason": None,
    }
    if not band_names:
        summary["reason"] = "no_bands"
        return [], [], summary

    stats_by_shape_name = {}
    unmapped_districts = []
    for gu_name, gu_stats in stats_by_gu.items():
        shape_name = SEOUL_GU_TO_ADM2_SHAPE.get(gu_name)
        if shape_name is None:
            unmapped_districts.append(gu_name)
            continue
        stats_by_shape_name[shape_name] = gu_stats

    if not stats_by_shape_name:
        summary["reason"] = "no_mapped_districts"
        summary["unmapped_districts"] = unmapped_districts
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
    matched_adm2_count = gu_stats_fc.size().getInfo()
    summary.update(
        {
            "matched_adm2_count": matched_adm2_count,
            "mapped_districts": sorted(stats_by_gu),
            "unmapped_districts": sorted(unmapped_districts),
        }
    )
    if matched_adm2_count == 0:
        summary["reason"] = "adm2_not_matched"
        return [], [], summary

    images = [
        gu_stats_fc.reduceToImage([band_name], ee.Reducer.first())
        .unmask(0)
        .rename(band_name)
        .clip(seoul)
        .reproject(crs=dem.projection(), scale=ANALYSIS_SCALE)
        for band_name in band_names
    ]
    summary["used"] = True
    return images, band_names, summary


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
region_boundary = seoul_fc.geometry()
analysis_region = (
    region_boundary.buffer(BOUNDARY_BUFFER_M)
    if BOUNDARY_BUFFER_M > 0
    else region_boundary
)
# 기존 분석 코드는 seoul 변수를 분석 영역으로 사용한다. 향후 다른 지역으로
# 확장할 때도 region_boundary와 analysis_region만 바꾸면 같은 흐름을 재사용할 수 있다.
seoul = analysis_region
print(
    "Analysis boundary:",
    {
        "boundary_buffer_m": BOUNDARY_BUFFER_M,
        "reference_points_filtered_to_analysis_boundary": True,
    },
)

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
print(
    "Model evaluation mode:",
    {
        "run_full_cv": RUN_FULL_CV,
        "evaluation_folds": EVALUATION_FOLDS,
        "feature_ablations": RUN_FEATURE_ABLATIONS,
        "water_feature_ablations": RUN_WATER_FEATURE_ABLATIONS,
        "drainage_ablation": RUN_DRAINAGE_ABLATION,
        "alpha_ablation": RUN_ALPHA_ABLATION,
        "hotspot_eval": RUN_HOTSPOT_EVAL,
        "hotspot_eval_in_cv": HOTSPOT_EVAL_IN_CV,
        "hotspot_eval_percentiles": HOTSPOT_EVAL_PERCENTILES,
        "risk_grade_percentiles": RISK_GRADE_PERCENTILES,
        "generate_map_outputs": GENERATE_MAP_OUTPUTS,
        "drainage_feature_mode": DRAINAGE_FEATURE_MODE,
        "drainage_infra_enabled": DRAINAGE_INFRA_ENABLED,
        "drainage_infra_radius_m": DRAINAGE_INFRA_RADIUS_M,
        "drainage_infra_min_active_points": DRAINAGE_INFRA_MIN_ACTIVE_POINTS,
        "drainage_gu_stats_enabled": DRAINAGE_GU_STATS_ENABLED,
        "drainage_gu_stats_min_districts": DRAINAGE_GU_STATS_MIN_DISTRICTS,
        "coverage_diagnostics": RUN_COVERAGE_DIAGNOSTICS,
        "buffer_sensitivity": RUN_BUFFER_SENSITIVITY,
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
all_positive_points = add_spatial_fold(ee.FeatureCollection(positive_features))
positive_points = all_positive_points.filterBounds(analysis_region)
positive_geom = positive_points.geometry()
train_positive_points = positive_points.filter(ee.Filter.neq("fold", VALIDATION_FOLD))
valid_positive_points = positive_points.filter(ee.Filter.eq("fold", VALIDATION_FOLD))
train_positive_geom = train_positive_points.geometry()
all_positive_count = all_positive_points.size().getInfo()
analysis_positive_count = positive_points.size().getInfo()
excluded_positive_count = all_positive_count - analysis_positive_count
train_positive_count = train_positive_points.size().getInfo()
valid_positive_count = valid_positive_points.size().getInfo()
positive_fold_hist = positive_points.aggregate_histogram("fold").getInfo()
print("Positive flood points:", all_positive_count)
print(
    "Positive points inside analysis boundary:",
    {
        "included": analysis_positive_count,
        "excluded": excluded_positive_count,
        "boundary_buffer_m": BOUNDARY_BUFFER_M,
    },
)
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

# 기존 수역/최근 수역까지의 거리다. 물 자체 여부보다 주변 하천·수계 접근성을
# 표현하기 위한 후보 feature로 사용한다.
water_source = water_occ.gte(0.2).Or(dw_water.gte(0.25)).unmask(0).rename("water_source")
water_dist_m = (
    water_source.fastDistanceTransform(WATER_DISTANCE_PIXELS)
    .sqrt()
    .multiply(ANALYSIS_SCALE)
    .rename("water_dist_m")
    .clip(seoul)
)
print("Water distance candidate max pixels:", WATER_DISTANCE_PIXELS)

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

# 데이터 양이 충분한 빗물관리 인프라만 후보 feature로 사용한다.
# 작은 시설별 파일은 레코드 수가 적어 우선 제외하고, 밀도가 의미 있게 계산되는
# 저영향개발 사전협의 지점과 빗물이용시설 지점만 1차 후보로 둔다.
drainage_infra_summary = []
drainage_feature_images = []
drainage_feature_bands = []
drainage_point_feature_bands = []
drainage_gu_feature_bands = []
if DRAINAGE_INFRA_ENABLED:
    drainage_sources = [
        {
            "dataset": "lid_preconsult",
            "path": LID_PRECONSULT_ZIP,
            "band": "lid_preconsult_density",
        },
        {
            "dataset": "rainwater_use",
            "path": RAINWATER_USE_ZIP,
            "band": "rainwater_use_density",
        },
    ]
    for source in drainage_sources:
        point_fc, source_summary = load_epsg5186_point_zip(
            source["path"],
            source["dataset"],
            DRAINAGE_INFRA_MIN_ACTIVE_POINTS,
        )
        if source_summary["used"]:
            inside_count = point_fc.filterBounds(seoul).size().getInfo()
            source_summary["points_in_analysis_boundary"] = inside_count
            if inside_count < DRAINAGE_INFRA_MIN_ACTIVE_POINTS:
                source_summary["used"] = False
                source_summary["reason"] = "too_few_points_inside_analysis_boundary"
                drainage_infra_summary.append(source_summary)
                continue
            density_image, density_p95 = make_point_density_feature(
                point_fc,
                source["band"],
                DRAINAGE_INFRA_RADIUS_M,
            )
            drainage_feature_images.append(density_image)
            drainage_feature_bands.append(source["band"])
            drainage_point_feature_bands.append(source["band"])
            source_summary.update(
                {
                    "band": source["band"],
                    "density_radius_m": DRAINAGE_INFRA_RADIUS_M,
                    "density_p95": density_p95,
                }
            )
        drainage_infra_summary.append(source_summary)
else:
    drainage_infra_summary.append(
        {
            "dataset": "drainage_infra",
            "used": False,
            "reason": "disabled",
        }
    )
if DRAINAGE_GU_STATS_ENABLED:
    gu_stat_sources = [
        load_pump_station_gu_stats(
            PUMP_STATION_CSV,
            DRAINAGE_GU_STATS_MIN_DISTRICTS,
        ),
        load_sewer_sensor_gu_stats(
            SEWER_SENSOR_GU_STATS_CSV,
            SEWER_LEVEL_SENSOR_CSV,
            DRAINAGE_GU_STATS_MIN_DISTRICTS,
        ),
    ]
    for stats_by_gu, source_summary in gu_stat_sources:
        if source_summary["used"]:
            stat_images, stat_bands, image_summary = make_gu_stat_feature_images(
                stats_by_gu,
                source_summary["dataset"],
            )
            source_summary["image_summary"] = image_summary
            if image_summary["used"]:
                drainage_feature_images.extend(stat_images)
                drainage_feature_bands.extend(stat_bands)
                drainage_gu_feature_bands.extend(stat_bands)
            else:
                source_summary["used"] = False
                source_summary["reason"] = image_summary["reason"]
        drainage_infra_summary.append(source_summary)
else:
    drainage_infra_summary.append(
        {
            "dataset": "drainage_gu_stats",
            "used": False,
            "reason": "disabled",
        }
    )
print("Drainage/rain-management infrastructure summary:")
for row in drainage_infra_summary:
    print(row)

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
static_feature_image = ee.Image.cat(
    [slope, hnd, log_upa, water_occ, built, lowland, water_dist_m]
    + drainage_feature_images
)
base_static_feature_bands = ["slope", "hnd", "log_upa", "water_occ", "built", "lowland"]
water_distance_feature_bands = ["water_dist_m"]
alpha_feature_bands = ["alpha_score"]
hybrid_feature_bands = base_static_feature_bands + alpha_feature_bands
drainage_candidate_feature_bands = hybrid_feature_bands + drainage_feature_bands
drainage_point_candidate_feature_bands = hybrid_feature_bands + drainage_point_feature_bands
drainage_gu_candidate_feature_bands = hybrid_feature_bands + drainage_gu_feature_bands
print("Fixed hybrid model bands:", hybrid_feature_bands)
print("Drainage candidate bands:", drainage_feature_bands)
print("Drainage point-density candidate bands:", drainage_point_feature_bands)
print("Drainage gu-stat candidate bands:", drainage_gu_feature_bands)
print("Candidate static feature bands:", static_feature_image.bandNames().getInfo())

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


def sample_split(feature_image, input_bands, positive_mask, negative_mask, area_mask, seed):
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


def compute_hotspot_metrics(probability, validation_points):
    """위험도 상위 구간이 검증 침수 기준점을 얼마나 포함하는지 계산한다."""
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
        evaluated_point_recall_value = ee.Number(
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
        evaluated_point_recall = ee.Algorithms.If(
            sampled_count.gt(0),
            evaluated_point_recall_value,
            None,
        )
        all_point_recall = ee.Algorithms.If(
            valid_count.gt(0),
            all_point_recall_value,
            None,
        )
        sample_coverage = ee.Algorithms.If(
            valid_count.gt(0),
            ee.Number(sampled_count).divide(valid_count),
            None,
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
        hotspot_area_share_value = ee.Number(
            ee.Algorithms.If(
                analysis_area.gt(0),
                hotspot_area.divide(analysis_area),
                0,
            )
        )
        hotspot_area_share = ee.Algorithms.If(
            analysis_area.gt(0),
            hotspot_area_share_value,
            None,
        )
        evaluated_point_lift = ee.Algorithms.If(
            hotspot_area_share_value.gt(0),
            evaluated_point_recall_value.divide(hotspot_area_share_value),
            None,
        )
        all_point_lift = ee.Algorithms.If(
            hotspot_area_share_value.gt(0),
            all_point_recall_value.divide(hotspot_area_share_value),
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
                    "evaluated_positive_points": sampled_count,
                    "excluded_positive_points": excluded_count,
                    "sample_coverage": sample_coverage,
                    "hit_points": hit_count,
                    "point_recall": evaluated_point_recall,
                    "sampled_point_recall": evaluated_point_recall,
                    "evaluated_point_recall": evaluated_point_recall,
                    "all_point_recall": all_point_recall,
                    "hotspot_area_share": hotspot_area_share,
                    "evaluated_point_lift": evaluated_point_lift,
                    "all_point_lift": all_point_lift,
                    "analysis_area_km2": analysis_area,
                    "hotspot_area_km2": hotspot_area,
                },
            )
        )

    metrics = ee.FeatureCollection(metric_features).getInfo()["features"]
    return [feature["properties"] for feature in metrics]


def build_risk_grade(probability):
    """확률 분위수를 기준으로 1~N 단계 위험도 등급 영상을 만든다."""
    minmax_info = probability.reduceRegion(
        reducer=ee.Reducer.minMax(),
        geometry=seoul,
        scale=ANALYSIS_SCALE,
        maxPixels=1e9,
    ).getInfo()
    threshold_info = probability.reduceRegion(
        reducer=ee.Reducer.percentile(RISK_GRADE_PERCENTILES),
        geometry=seoul,
        scale=ANALYSIS_SCALE,
        maxPixels=1e9,
    ).getInfo()
    thresholds = {
        percentile: threshold_info[f"flood_prob_p{percentile}"]
        for percentile in RISK_GRADE_PERCENTILES
    }

    risk_grade = ee.Image.constant(1).rename("risk_grade").updateMask(probability.mask())
    for index, percentile in enumerate(RISK_GRADE_PERCENTILES, start=2):
        risk_grade = risk_grade.where(
            probability.gte(ee.Number(thresholds[percentile])),
            index,
        )
    risk_grade = risk_grade.rename("risk_grade")

    area_info = (
        ee.Image.pixelArea()
        .divide(1e6)
        .rename("area")
        .addBands(risk_grade)
        .reduceRegion(
            reducer=ee.Reducer.sum().group(groupField=1, groupName="grade"),
            geometry=seoul,
            scale=ANALYSIS_SCALE,
            maxPixels=1e9,
        )
        .getInfo()
    )
    raw_groups = area_info.get("groups", [])
    area_by_grade = {
        int(group["grade"]): group["sum"]
        for group in raw_groups
    }

    grade_summaries = []
    lower_percentile = 0
    for grade in range(1, len(RISK_GRADE_PERCENTILES) + 2):
        upper_percentile = (
            RISK_GRADE_PERCENTILES[grade - 1]
            if grade <= len(RISK_GRADE_PERCENTILES)
            else 100
        )
        lower_threshold = (
            thresholds.get(lower_percentile)
            if lower_percentile in thresholds
            else minmax_info["flood_prob_min"]
        )
        upper_threshold = (
            thresholds.get(upper_percentile)
            if upper_percentile in thresholds
            else minmax_info["flood_prob_max"]
        )
        grade_summaries.append(
            {
                "grade": grade,
                "percentile_range": f"p{lower_percentile}-p{upper_percentile}",
                "probability_min": lower_threshold,
                "probability_max": upper_threshold,
                "area_km2": area_by_grade.get(grade, 0),
            }
        )
        lower_percentile = upper_percentile

    return risk_grade, thresholds, grade_summaries


def histogram_count(histogram, grade):
    for key, value in histogram.items():
        if int(float(key)) == grade:
            return int(value)
    return 0


def compute_risk_grade_point_metrics(risk_grade, validation_points, grade_summaries):
    """위험도 등급별 검증 침수 기준점 포함률과 lift를 계산한다."""
    scored_points = risk_grade.sampleRegions(
        collection=validation_points,
        scale=ANALYSIS_SCALE,
        geometries=False,
        tileScale=4,
    ).filter(ee.Filter.notNull(["risk_grade"]))
    histogram = scored_points.aggregate_histogram("risk_grade").getInfo()
    valid_count = validation_points.size().getInfo()
    evaluated_count = scored_points.size().getInfo()
    excluded_count = valid_count - evaluated_count
    total_area = sum(row["area_km2"] for row in grade_summaries)

    grade_metrics = []
    for row in grade_summaries:
        grade = row["grade"]
        hit_count = histogram_count(histogram, grade)
        area_share = row["area_km2"] / total_area if total_area else 0
        evaluated_point_share = hit_count / evaluated_count if evaluated_count else 0
        all_point_share = hit_count / valid_count if valid_count else 0
        lift = evaluated_point_share / area_share if area_share else None
        grade_metrics.append(
            {
                **row,
                "area_share": area_share,
                "valid_positive_points": valid_count,
                "evaluated_positive_points": evaluated_count,
                "excluded_positive_points": excluded_count,
                "hit_points": hit_count,
                "evaluated_point_share": evaluated_point_share,
                "all_point_share": all_point_share,
                "lift": lift,
            }
        )

    cumulative_metrics = []
    for min_grade in range(len(grade_summaries), 0, -1):
        included_rows = [
            row for row in grade_metrics if row["grade"] >= min_grade
        ]
        area_km2 = sum(row["area_km2"] for row in included_rows)
        hit_count = sum(row["hit_points"] for row in included_rows)
        area_share = area_km2 / total_area if total_area else 0
        evaluated_point_recall = hit_count / evaluated_count if evaluated_count else 0
        all_point_recall = hit_count / valid_count if valid_count else 0
        lift = evaluated_point_recall / area_share if area_share else None
        cumulative_metrics.append(
            {
                "grade_range": f"{min_grade}+",
                "min_grade": min_grade,
                "area_km2": area_km2,
                "area_share": area_share,
                "valid_positive_points": valid_count,
                "evaluated_positive_points": evaluated_count,
                "excluded_positive_points": excluded_count,
                "hit_points": hit_count,
                "evaluated_point_recall": evaluated_point_recall,
                "all_point_recall": all_point_recall,
                "lift": lift,
            }
        )

    return grade_metrics, cumulative_metrics


def build_risk_grade_legend(grade_summaries):
    """위험도 등급 지도에 사용할 HTML 범례 항목을 만든다."""
    legend = {}
    for row in grade_summaries:
        grade = row["grade"]
        label = f"Grade {grade}: {RISK_GRADE_NAMES[grade - 1]} ({row['percentile_range']})"
        legend[label] = RISK_GRADE_PALETTE[grade - 1]
    return legend


def inject_static_legend(html_path, title, legend_dict):
    """geemap widget 범례 대신 브라우저에서 바로 보이는 정적 HTML 범례를 삽입한다."""
    items_html = "\n".join(
        (
            '<div class="risk-grade-legend-item">'
            f'<span class="risk-grade-legend-swatch" style="background:{html.escape(color, quote=True)}"></span>'
            f'<span>{html.escape(label)}</span>'
            "</div>"
        )
        for label, color in legend_dict.items()
    )
    legend_html = f"""
<style>
.risk-grade-legend {{
  position: fixed;
  right: 18px;
  bottom: 28px;
  z-index: 999999;
  max-width: 260px;
  padding: 12px 14px;
  border: 1px solid rgba(0, 0, 0, 0.18);
  border-radius: 6px;
  background: rgba(255, 255, 255, 0.94);
  box-shadow: 0 2px 12px rgba(0, 0, 0, 0.20);
  color: #222;
  font: 13px/1.35 Arial, sans-serif;
}}
.risk-grade-legend-title {{
  margin-bottom: 8px;
  font-weight: 700;
}}
.risk-grade-legend-item {{
  display: flex;
  align-items: center;
  gap: 8px;
  margin: 4px 0;
}}
.risk-grade-legend-swatch {{
  width: 16px;
  height: 16px;
  border: 1px solid rgba(0, 0, 0, 0.28);
  flex: 0 0 16px;
}}
</style>
<div class="risk-grade-legend">
  <div class="risk-grade-legend-title">{html.escape(title)}</div>
  {items_html}
</div>
"""
    with open(html_path, "r", encoding="utf-8") as f:
        content = f.read()

    if "</body>" in content:
        content = content.replace("</body>", f"{legend_html}\n</body>", 1)
    else:
        content = f"{content}\n{legend_html}"

    with open(html_path, "w", encoding="utf-8") as f:
        f.write(content)


def sanitize_exported_widget_controls(html_path):
    """정적 HTML에서 깨질 수 있는 빈 geemap widget control을 제거한다."""
    with open(html_path, "r", encoding="utf-8") as f:
        content = f.read()

    pattern = re.compile(
        r'(<script type="application/vnd\.jupyter\.widget-state\+json">\s*)'
        r"({.*?})"
        r"(\s*</script>)",
        re.DOTALL,
    )
    match = pattern.search(content)
    if not match:
        return

    state = json.loads(match.group(2))
    models = state.get("state", {})
    remove_model_ids = set()
    for model_id, model in models.items():
        if model.get("model_name") == "LeafletWidgetControlModel":
            remove_model_ids.add(model_id)
            widget_ref = model.get("state", {}).get("widget")
            if isinstance(widget_ref, str) and widget_ref.startswith("IPY_MODEL_"):
                remove_model_ids.add(widget_ref.replace("IPY_MODEL_", "", 1))

    for model_id in list(remove_model_ids):
        model = models.get(model_id)
        if not model:
            continue
        layout_ref = model.get("state", {}).get("layout")
        if isinstance(layout_ref, str) and layout_ref.startswith("IPY_MODEL_"):
            remove_model_ids.add(layout_ref.replace("IPY_MODEL_", "", 1))

    if not remove_model_ids:
        return

    for model_id in remove_model_ids:
        models.pop(model_id, None)

    removed_refs = {f"IPY_MODEL_{model_id}" for model_id in remove_model_ids}
    for model in models.values():
        controls = model.get("state", {}).get("controls")
        if isinstance(controls, list):
            model["state"]["controls"] = [
                control for control in controls if control not in removed_refs
            ]

    updated_state = json.dumps(state, ensure_ascii=False)
    content = (
        content[: match.start()]
        + match.group(1)
        + updated_state
        + match.group(3)
        + content[match.end():]
    )

    with open(html_path, "w", encoding="utf-8") as f:
        f.write(content)


def diagnose_validation_coverage(feature_image, input_bands, probability, validation_points):
    """검증 기준점에서 feature/probability mask가 생기는 원인을 요약한다."""
    validity_bands = [
        feature_image.select(band).mask().rename(f"{band}_valid")
        for band in input_bands
    ]
    expected_count = validation_points.size().getInfo()
    input_validity = ee.Image.cat(validity_bands).unmask(0, False)
    all_inputs_valid = input_validity.reduce(ee.Reducer.min()).rename("all_inputs_valid")
    probability_valid = probability.mask().rename("flood_prob_valid").unmask(0, False)
    seoul_pixel = (
        ee.Image.constant(1)
        .clip(seoul)
        .mask()
        .rename("inside_seoul_pixel")
        .unmask(0, False)
    )
    diagnostics_image = ee.Image.cat(
        [
            input_validity,
            all_inputs_valid,
            probability_valid,
            seoul_pixel,
            ee.Image.pixelLonLat(),
        ]
    )
    diagnostics = diagnostics_image.sampleRegions(
        collection=validation_points,
        scale=ANALYSIS_SCALE,
        geometries=True,
        tileScale=4,
    ).getInfo()["features"]

    band_valid_counts = {
        band: sum(
            1
            for feature in diagnostics
            if feature["properties"].get(f"{band}_valid", 0) >= 1
        )
        for band in input_bands
    }
    full_input_count = sum(
        1
        for feature in diagnostics
        if feature["properties"].get("all_inputs_valid", 0) >= 1
    )
    probability_count = sum(
        1
        for feature in diagnostics
        if feature["properties"].get("flood_prob_valid", 0) >= 1
    )
    inside_seoul_pixel_count = sum(
        1
        for feature in diagnostics
        if feature["properties"].get("inside_seoul_pixel", 0) >= 1
    )

    missing_points = []
    for feature in diagnostics:
        props = feature["properties"]
        if (
            props.get("all_inputs_valid", 0) >= 1
            and props.get("flood_prob_valid", 0) >= 1
        ):
            continue

        lon, lat = feature["geometry"]["coordinates"]
        missing_bands = [
            band
            for band in input_bands
            if props.get(f"{band}_valid", 0) < 1
        ]
        missing_points.append(
            {
                "lon": lon,
                "lat": lat,
                "inside_seoul_pixel": props.get("inside_seoul_pixel", 0),
                "flood_prob_valid": props.get("flood_prob_valid", 0),
                "missing_bands": missing_bands,
            }
        )

    return {
        "validation_points": expected_count,
        "diagnostic_samples": len(diagnostics),
        "inside_seoul_pixel_count": inside_seoul_pixel_count,
        "all_input_bands_valid_count": full_input_count,
        "probability_valid_count": probability_count,
        "band_valid_counts": band_valid_counts,
        "missing_points": missing_points,
    }


def run_hybrid_fold(
    validation_fold,
    positive_buffer_m=POSITIVE_BUFFER_M,
    negative_buffer_m=NEGATIVE_BUFFER_M,
    include_map_outputs=False,
    include_importance=False,
    include_hotspot_metrics=False,
    input_bands=None,
):
    if input_bands is None:
        input_bands = hybrid_feature_bands

    fold_inputs = build_hybrid_inputs(validation_fold, positive_buffer_m)
    negative_mask = make_negative_mask(negative_buffer_m)
    train_fc = sample_split(
        fold_inputs["feature_image"],
        input_bands,
        fold_inputs["positive_train_mask"],
        negative_mask,
        fold_inputs["train_area_mask"],
        700 + validation_fold,
    )
    valid_fc = sample_split(
        fold_inputs["feature_image"],
        input_bands,
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

    if include_map_outputs or include_hotspot_metrics:
        probability = (
            fold_inputs["feature_image"]
            .select(input_bands)
            .classify(classifier, "flood_prob")
            .rename("flood_prob")
            .clip(seoul)
        )

        if include_hotspot_metrics:
            result["hotspot_metrics"] = compute_hotspot_metrics(
                probability,
                fold_inputs["valid_positive"],
            )

    if include_map_outputs:
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


def sanitize_fold_result(row):
    """EE 객체를 제외하고 fold 평가 결과의 표/JSON 저장용 값만 남긴다."""
    result = {
        "fold": row["fold"],
        "positive_buffer_m": row["positive_buffer_m"],
        "negative_buffer_m": row["negative_buffer_m"],
        "train_positive_count": row["train_positive_count"],
        "valid_positive_count": row["valid_positive_count"],
        "train_sample_count": row["train_sample_count"],
        "valid_sample_count": row["valid_sample_count"],
        "train_label_histogram": row["train_label_histogram"],
        "valid_label_histogram": row["valid_label_histogram"],
        "train_confusion": row["train_confusion"],
        "valid_confusion": row["valid_confusion"],
        "valid_accuracy": row["valid_accuracy"],
        "valid_kappa": row["valid_kappa"],
    }
    if "hotspot_metrics" in row:
        result["hotspot_point_recall"] = compact_hotspot_recall(
            row["hotspot_metrics"]
        )
    return result


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


def compact_hotspot_recall(metrics):
    return {
        f"top_{int(metric['top_percent'])}_percent": metric["evaluated_point_recall"]
        for metric in metrics
    }


def summarize_hotspot_metrics(rows):
    """fold별 Top-k hotspot 포함률의 평균과 표준편차를 요약한다."""
    grouped = {}
    for row in rows:
        for metric in row.get("hotspot_metrics", []):
            percentile = int(metric["percentile"])
            if percentile not in grouped:
                grouped[percentile] = {
                    "percentile": percentile,
                    "top_percent": int(metric["top_percent"]),
                    "point_recalls": [],
                    "all_point_recalls": [],
                    "point_lifts": [],
                    "area_shares": [],
                    "hotspot_areas": [],
                    "hit_points": [],
                }
            grouped[percentile]["point_recalls"].append(metric["evaluated_point_recall"])
            grouped[percentile]["all_point_recalls"].append(metric["all_point_recall"])
            grouped[percentile]["point_lifts"].append(metric["evaluated_point_lift"])
            grouped[percentile]["area_shares"].append(metric["hotspot_area_share"])
            grouped[percentile]["hotspot_areas"].append(metric["hotspot_area_km2"])
            grouped[percentile]["hit_points"].append(metric["hit_points"])

    summary = []
    for percentile in sorted(grouped):
        values = grouped[percentile]
        summary.append(
            {
                "percentile": values["percentile"],
                "top_percent": values["top_percent"],
                "evaluated_point_recall_mean": mean(values["point_recalls"]),
                "evaluated_point_recall_std": sample_std(values["point_recalls"]),
                "all_point_recall_mean": mean(values["all_point_recalls"]),
                "evaluated_point_lift_mean": mean(values["point_lifts"]),
                "hotspot_area_share_mean": mean(values["area_shares"]),
                "hit_points_mean": mean(values["hit_points"]),
                "hotspot_area_km2_mean": mean(values["hotspot_areas"]),
            }
        )
    return summary


def print_validation_delta(label, baseline_summary, comparison_summary):
    print(
        label,
        {
            "baseline_accuracy_mean": baseline_summary["accuracy_mean"],
            "comparison_accuracy_mean": comparison_summary["accuracy_mean"],
            "delta_accuracy": comparison_summary["accuracy_mean"] - baseline_summary["accuracy_mean"],
            "baseline_kappa_mean": baseline_summary["kappa_mean"],
            "comparison_kappa_mean": comparison_summary["kappa_mean"],
            "delta_kappa": comparison_summary["kappa_mean"] - baseline_summary["kappa_mean"],
        },
    )


def print_hotspot_delta(label, baseline_summary, comparison_summary):
    baseline_by_percentile = {
        row["percentile"]: row for row in baseline_summary
    }
    comparison_by_percentile = {
        row["percentile"]: row for row in comparison_summary
    }
    common_percentiles = sorted(
        set(baseline_by_percentile).intersection(comparison_by_percentile)
    )
    if not common_percentiles:
        return

    print(label)
    for percentile in common_percentiles:
        baseline = baseline_by_percentile[percentile]
        comparison = comparison_by_percentile[percentile]
        print(
            {
                "percentile": percentile,
                "top_percent": baseline["top_percent"],
                "baseline_recall": baseline["evaluated_point_recall_mean"],
                "comparison_recall": comparison["evaluated_point_recall_mean"],
                "delta_recall": (
                    comparison["evaluated_point_recall_mean"]
                    - baseline["evaluated_point_recall_mean"]
                ),
                "baseline_lift": baseline["evaluated_point_lift_mean"],
                "comparison_lift": comparison["evaluated_point_lift_mean"],
                "delta_lift": (
                    comparison["evaluated_point_lift_mean"]
                    - baseline["evaluated_point_lift_mean"]
                ),
            }
        )


def save_experiment_outputs(metrics_payload, csv_tables):
    write_json(METRICS_JSON, metrics_payload)
    for path, rows in csv_tables.items():
        write_csv(path, rows)
    print(
        "Saved metrics outputs:",
        {
            "metrics_json": METRICS_JSON,
            **{os.path.basename(path): path for path in csv_tables},
        },
    )


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


def run_spatial_cv(
    model_name,
    input_bands,
    include_importance=False,
    include_hotspot_metrics=False,
    folds=None,
):
    folds = EVALUATION_FOLDS if folds is None else folds
    label = (
        "spatial cross-validation"
        if len(folds) > 1
        else "quick spatial validation"
    )
    print(f"\n{model_name} {label}:")
    rows = []
    for fold in folds:
        result = run_hybrid_fold(
            fold,
            include_map_outputs=False,
            include_importance=include_importance,
            include_hotspot_metrics=include_hotspot_metrics,
            input_bands=input_bands,
        )
        rows.append(result)
        fold_log = {
            "fold": result["fold"],
            "train_pos": result["train_positive_count"],
            "valid_pos": result["valid_positive_count"],
            "valid_accuracy": result["valid_accuracy"],
            "valid_kappa": result["valid_kappa"],
            "valid_confusion": result["valid_confusion"],
        }
        if "hotspot_metrics" in result:
            fold_log["hotspot_point_recall"] = compact_hotspot_recall(
                result["hotspot_metrics"]
            )
        print(fold_log)
    return rows


cv_results = run_spatial_cv(
    "Hybrid-basic",
    hybrid_feature_bands,
    include_importance=True,
    include_hotspot_metrics=HOTSPOT_EVAL_IN_CV,
)

cv_summary = summarize_validation(cv_results)
print("Hybrid-basic validation summary:", cv_summary)
importance_summary = summarize_importance(cv_results, hybrid_feature_bands)
print("Hybrid-basic normalized feature importance summary:")
for row in importance_summary:
    print(row)
hotspot_cv_summary = summarize_hotspot_metrics(cv_results)
if hotspot_cv_summary:
    print("Hybrid-basic hotspot recall summary:")
    for row in hotspot_cv_summary:
        print(row)

no_alpha_cv_results = []
no_alpha_cv_summary = None
no_alpha_hotspot_summary = []
if RUN_ALPHA_ABLATION:
    no_alpha_cv_results = run_spatial_cv(
        "Hybrid-no-alpha",
        base_static_feature_bands,
        include_importance=False,
        include_hotspot_metrics=HOTSPOT_EVAL_IN_CV,
    )
    no_alpha_cv_summary = summarize_validation(no_alpha_cv_results)
    print("Hybrid-no-alpha validation summary:", no_alpha_cv_summary)
    print_validation_delta(
        "AlphaEarth ablation validation delta:",
        cv_summary,
        no_alpha_cv_summary,
    )
    no_alpha_hotspot_summary = summarize_hotspot_metrics(no_alpha_cv_results)
    if no_alpha_hotspot_summary:
        print("Hybrid-no-alpha hotspot recall summary:")
        for row in no_alpha_hotspot_summary:
            print(row)
        print_hotspot_delta(
            "AlphaEarth ablation hotspot recall/lift delta:",
            hotspot_cv_summary,
            no_alpha_hotspot_summary,
        )

selected_model_name = "Hybrid-basic"
selected_feature_bands = hybrid_feature_bands
selected_kappa_mean = cv_summary["kappa_mean"]
no_water_cv_results = []
no_water_cv_summary = None
water_dist_cv_results = []
water_dist_cv_summary = None
drainage_cv_results_by_model = {}
drainage_cv_summaries = {}

if not RUN_FEATURE_ABLATIONS:
    if DRAINAGE_FEATURE_MODE == "points" and drainage_point_feature_bands:
        selected_model_name = "Hybrid-plus-drainage-points"
        selected_feature_bands = drainage_point_candidate_feature_bands
    elif DRAINAGE_FEATURE_MODE == "gu_stats" and drainage_gu_feature_bands:
        selected_model_name = "Hybrid-plus-drainage-gu-stats"
        selected_feature_bands = drainage_gu_candidate_feature_bands
    elif DRAINAGE_FEATURE_MODE == "all" and drainage_feature_bands:
        selected_model_name = "Hybrid-plus-drainage-infra"
        selected_feature_bands = drainage_candidate_feature_bands
    if selected_model_name != "Hybrid-basic":
        print(
            "Default drainage feature mode selected:",
            {
                "mode": DRAINAGE_FEATURE_MODE,
                "model": selected_model_name,
                "bands": selected_feature_bands,
            },
        )

if RUN_FEATURE_ABLATIONS:
    if RUN_WATER_FEATURE_ABLATIONS:
        no_water_feature_bands = [
            band for band in hybrid_feature_bands if band != "water_occ"
        ]
        no_water_cv_results = run_spatial_cv(
            "Hybrid-no-water-occ",
            no_water_feature_bands,
            include_importance=False,
        )
        no_water_cv_summary = summarize_validation(no_water_cv_results)
        print("Hybrid-no-water-occ validation summary:", no_water_cv_summary)
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

        if no_water_cv_summary["kappa_mean"] >= selected_kappa_mean:
            selected_model_name = "Hybrid-no-water-occ"
            selected_feature_bands = no_water_feature_bands
            selected_kappa_mean = no_water_cv_summary["kappa_mean"]
            print("Selected feature set: water_occ removed from RF inputs.")
        else:
            print("Selected feature set: keeping water_occ in RF inputs.")

        water_dist_feature_bands = hybrid_feature_bands + water_distance_feature_bands
        water_dist_cv_results = run_spatial_cv(
            "Hybrid-plus-water-distance",
            water_dist_feature_bands,
            include_importance=False,
        )
        water_dist_cv_summary = summarize_validation(water_dist_cv_results)
        print("Hybrid-plus-water-distance validation summary:", water_dist_cv_summary)
        print(
            "Water distance ablation:",
            {
                "baseline_kappa_mean": cv_summary["kappa_mean"],
                "water_dist_kappa_mean": water_dist_cv_summary["kappa_mean"],
                "delta_kappa": water_dist_cv_summary["kappa_mean"] - cv_summary["kappa_mean"],
                "baseline_accuracy_mean": cv_summary["accuracy_mean"],
                "water_dist_accuracy_mean": water_dist_cv_summary["accuracy_mean"],
                "delta_accuracy": water_dist_cv_summary["accuracy_mean"] - cv_summary["accuracy_mean"],
            },
        )

        if water_dist_cv_summary["kappa_mean"] > selected_kappa_mean:
            selected_model_name = "Hybrid-plus-water-distance"
            selected_feature_bands = water_dist_feature_bands
            selected_kappa_mean = water_dist_cv_summary["kappa_mean"]
            print("Selected feature set: water_dist_m added to RF inputs.")
        else:
            print("Selected feature set: water_dist_m not added to RF inputs.")
    else:
        print("Water feature ablations skipped.")

    if RUN_DRAINAGE_ABLATION and drainage_feature_bands:
        drainage_candidate_configs = []
        if drainage_point_feature_bands:
            drainage_candidate_configs.append(
                {
                    "model": "Hybrid-plus-drainage-points",
                    "bands": drainage_point_candidate_feature_bands,
                }
            )
        if drainage_gu_feature_bands:
            drainage_candidate_configs.append(
                {
                    "model": "Hybrid-plus-drainage-gu-stats",
                    "bands": drainage_gu_candidate_feature_bands,
                }
            )
        if drainage_point_feature_bands and drainage_gu_feature_bands:
            drainage_candidate_configs.append(
                {
                    "model": "Hybrid-plus-drainage-infra",
                    "bands": drainage_candidate_feature_bands,
                }
            )

        for config in drainage_candidate_configs:
            model_name = config["model"]
            candidate_bands = config["bands"]
            candidate_results = run_spatial_cv(
                model_name,
                candidate_bands,
                include_importance=False,
            )
            candidate_summary = summarize_validation(candidate_results)
            drainage_cv_results_by_model[model_name] = candidate_results
            drainage_cv_summaries[model_name] = candidate_summary
            print(f"{model_name} validation summary:", candidate_summary)
            print(
                f"{model_name} validation delta:",
                {
                    "baseline_kappa_mean": cv_summary["kappa_mean"],
                    "candidate_kappa_mean": candidate_summary["kappa_mean"],
                    "delta_kappa": candidate_summary["kappa_mean"] - cv_summary["kappa_mean"],
                    "baseline_accuracy_mean": cv_summary["accuracy_mean"],
                    "candidate_accuracy_mean": candidate_summary["accuracy_mean"],
                    "delta_accuracy": candidate_summary["accuracy_mean"] - cv_summary["accuracy_mean"],
                },
            )

            if candidate_summary["kappa_mean"] > selected_kappa_mean:
                selected_model_name = model_name
                selected_feature_bands = candidate_bands
                selected_kappa_mean = candidate_summary["kappa_mean"]

        if selected_model_name.startswith("Hybrid-plus-drainage"):
            print("Selected feature set: drainage candidate added to RF inputs.")
        else:
            print("Selected feature set: drainage candidates not added to RF inputs.")
    elif RUN_DRAINAGE_ABLATION:
        print("Drainage infrastructure candidate skipped: no source passed data-volume filter.")
    else:
        print("Drainage infrastructure ablation skipped.")
else:
    print("\nFeature ablations skipped for fast model-building run.")
    print("Set RUN_FEATURE_ABLATIONS=1 to retest candidate feature sets.")

if RUN_BUFFER_SENSITIVITY:
    buffer_sensitivity_rows = run_buffer_sensitivity()
else:
    buffer_sensitivity_rows = []

hybrid_result = run_hybrid_fold(
    VALIDATION_FOLD,
    include_map_outputs=GENERATE_MAP_OUTPUTS,
    include_importance=True,
    include_hotspot_metrics=RUN_HOTSPOT_EVAL and GENERATE_MAP_OUTPUTS,
    input_bands=selected_feature_bands,
)
seoul_probability = None
alpha_score = hybrid_result["alpha_score"]
hotspots = None
threshold = None
prob_stats = None
hotspot_area = None
valid_area_mask = hybrid_result["valid_area_mask"]
valid_positive_points = hybrid_result["valid_positive"]
risk_grade = None
risk_grade_thresholds = {}
risk_grade_summaries = []
risk_grade_point_metrics = []
cumulative_risk_grade_point_metrics = []
risk_grade_legend = {}

print(f"\nSelected fold: {VALIDATION_FOLD}")
print("Selected model:", selected_model_name)
print("Selected bands:", selected_feature_bands)
print("Selected fold validation accuracy:", hybrid_result["valid_accuracy"])
print("Selected fold validation kappa:", hybrid_result["valid_kappa"])
selected_importance_summary = summarize_importance([hybrid_result], selected_feature_bands)
print("Selected model normalized feature importance summary:")
for row in selected_importance_summary:
    print(row)

if GENERATE_MAP_OUTPUTS:
    seoul_probability = hybrid_result["probability"]
    hotspots = hybrid_result["hotspots"]
    threshold = hybrid_result["threshold"]
    prob_stats = hybrid_result["prob_stats"]
    hotspot_area = hybrid_result["hotspot_area"]
    risk_grade, risk_grade_thresholds, risk_grade_summaries = build_risk_grade(seoul_probability)
    risk_grade_point_metrics, cumulative_risk_grade_point_metrics = (
        compute_risk_grade_point_metrics(
            risk_grade,
            valid_positive_points,
            risk_grade_summaries,
        )
    )
    risk_grade_legend = build_risk_grade_legend(risk_grade_summaries)

    print("Seoul probability stats:", prob_stats)
    print(f"P{HOTSPOT_PERCENTILE} threshold:", threshold)
    print("Hotspot area (km²):", hotspot_area)
    print("Risk grade percentile thresholds:", risk_grade_thresholds)
    print("Risk grade area summary:")
    for row in risk_grade_summaries:
        print(row)
    print("Risk grade legend:", risk_grade_legend)
    print("Risk grade validation point summary:")
    for row in risk_grade_point_metrics:
        print(row)
    print("Cumulative high-risk grade validation point summary:")
    for row in cumulative_risk_grade_point_metrics:
        print(row)
    if "hotspot_metrics" in hybrid_result:
        print("Selected fold hotspot recall metrics:")
        for row in hybrid_result["hotspot_metrics"]:
            print(row)
    if RUN_COVERAGE_DIAGNOSTICS:
        coverage_diagnostics = diagnose_validation_coverage(
            hybrid_result["feature_image"],
            selected_feature_bands,
            seoul_probability,
            valid_positive_points,
        )
        print("Selected fold validation coverage diagnostics:")
        print(
            {
                "validation_points": coverage_diagnostics["validation_points"],
                "diagnostic_samples": coverage_diagnostics["diagnostic_samples"],
                "inside_seoul_pixel_count": coverage_diagnostics["inside_seoul_pixel_count"],
                "all_input_bands_valid_count": coverage_diagnostics["all_input_bands_valid_count"],
                "probability_valid_count": coverage_diagnostics["probability_valid_count"],
                "band_valid_counts": coverage_diagnostics["band_valid_counts"],
            }
        )
        if coverage_diagnostics["missing_points"]:
            print("Validation points without full prediction coverage:")
            for row in coverage_diagnostics["missing_points"]:
                print(row)
else:
    print("Map/risk-grade outputs skipped: GENERATE_MAP_OUTPUTS=0")

cv_result_rows = [
    {"model": "Hybrid-basic", **sanitize_fold_result(row)}
    for row in cv_results
]
if no_alpha_cv_results:
    cv_result_rows.extend(
        {"model": "Hybrid-no-alpha", **sanitize_fold_result(row)}
        for row in no_alpha_cv_results
    )
if no_water_cv_results:
    cv_result_rows.extend(
        {"model": "Hybrid-no-water-occ", **sanitize_fold_result(row)}
        for row in no_water_cv_results
    )
if water_dist_cv_results:
    cv_result_rows.extend(
        {"model": "Hybrid-plus-water-distance", **sanitize_fold_result(row)}
        for row in water_dist_cv_results
    )
for model_name, result_rows in drainage_cv_results_by_model.items():
    cv_result_rows.extend(
        {"model": model_name, **sanitize_fold_result(row)}
        for row in result_rows
    )

topk_summary_rows = [
    {"model": "Hybrid-basic", **row}
    for row in hotspot_cv_summary
]
if no_alpha_hotspot_summary:
    topk_summary_rows.extend(
        {"model": "Hybrid-no-alpha", **row}
        for row in no_alpha_hotspot_summary
    )

metrics_payload = {
    "config": {
        "year": YEAR,
        "analysis_scale": ANALYSIS_SCALE,
        "boundary_buffer_m": BOUNDARY_BUFFER_M,
        "reference_point_limit": REFERENCE_POINT_LIMIT,
        "positive_sample_points": POSITIVE_SAMPLE_POINTS,
        "negative_points": NEGATIVE_POINTS,
        "positive_buffer_m": POSITIVE_BUFFER_M,
        "negative_buffer_m": NEGATIVE_BUFFER_M,
        "hotspot_percentile": HOTSPOT_PERCENTILE,
        "hotspot_eval_percentiles": HOTSPOT_EVAL_PERCENTILES,
        "risk_grade_percentiles": RISK_GRADE_PERCENTILES,
        "spatial_block_degrees": SPATIAL_BLOCK_DEGREES,
        "spatial_folds": SPATIAL_FOLDS,
        "validation_fold": VALIDATION_FOLD,
        "evaluation_folds": EVALUATION_FOLDS,
        "run_full_cv": RUN_FULL_CV,
        "run_feature_ablations": RUN_FEATURE_ABLATIONS,
        "run_water_feature_ablations": RUN_WATER_FEATURE_ABLATIONS,
        "run_drainage_ablation": RUN_DRAINAGE_ABLATION,
        "run_alpha_ablation": RUN_ALPHA_ABLATION,
        "run_hotspot_eval": RUN_HOTSPOT_EVAL,
        "hotspot_eval_in_cv": HOTSPOT_EVAL_IN_CV,
        "generate_map_outputs": GENERATE_MAP_OUTPUTS,
        "run_buffer_sensitivity": RUN_BUFFER_SENSITIVITY,
        "water_distance_pixels": WATER_DISTANCE_PIXELS,
        "drainage_feature_mode": DRAINAGE_FEATURE_MODE,
        "drainage_infra_enabled": DRAINAGE_INFRA_ENABLED,
        "drainage_infra_radius_m": DRAINAGE_INFRA_RADIUS_M,
        "drainage_infra_min_active_points": DRAINAGE_INFRA_MIN_ACTIVE_POINTS,
        "drainage_gu_stats_enabled": DRAINAGE_GU_STATS_ENABLED,
        "drainage_gu_stats_min_districts": DRAINAGE_GU_STATS_MIN_DISTRICTS,
        "lid_preconsult_zip": LID_PRECONSULT_ZIP,
        "rainwater_use_zip": RAINWATER_USE_ZIP,
        "pump_station_csv": PUMP_STATION_CSV,
        "sewer_sensor_gu_stats_csv": SEWER_SENSOR_GU_STATS_CSV,
        "sewer_level_sensor_csv": SEWER_LEVEL_SENSOR_CSV,
        "risk_grade_palette": RISK_GRADE_PALETTE,
        "risk_grade_names": RISK_GRADE_NAMES,
    },
    "data": {
        "reference_geojson": SEOUL_REFERENCE_GEOJSON,
        "reference_points_total": all_positive_count,
        "reference_points_in_analysis_boundary": analysis_positive_count,
        "reference_points_excluded": excluded_positive_count,
        "positive_fold_histogram": positive_fold_hist,
        "selected_train_positive_count": train_positive_count,
        "selected_validation_positive_count": valid_positive_count,
        "drainage_infra": drainage_infra_summary,
        "drainage_feature_groups": {
            "point_density": drainage_point_feature_bands,
            "gu_stats": drainage_gu_feature_bands,
            "all": drainage_feature_bands,
        },
    },
    "selected_model": {
        "name": selected_model_name,
        "bands": selected_feature_bands,
        "validation_fold": VALIDATION_FOLD,
        "selected_fold_accuracy": hybrid_result["valid_accuracy"],
        "selected_fold_kappa": hybrid_result["valid_kappa"],
        "probability_stats": prob_stats,
        "hotspot_threshold": threshold,
        "hotspot_area": hotspot_area,
        "feature_importance": selected_importance_summary,
    },
    "validation": {
        "summary": cv_summary,
        "fold_results": cv_result_rows,
    },
    "feature_importance": importance_summary,
    "topk": {
        "cv_summary": topk_summary_rows,
        "selected_fold": hybrid_result.get("hotspot_metrics", []),
    },
    "risk_grade": {
        "thresholds": risk_grade_thresholds,
        "area_summary": risk_grade_summaries,
        "point_summary": risk_grade_point_metrics,
        "cumulative_point_summary": cumulative_risk_grade_point_metrics,
        "legend": risk_grade_legend,
    },
    "alpha_ablation": {
        "validation_summary": no_alpha_cv_summary,
        "hotspot_summary": no_alpha_hotspot_summary,
    } if no_alpha_cv_summary else None,
    "feature_ablations": {
        "no_water_occ": no_water_cv_summary,
        "water_distance": water_dist_cv_summary,
        "drainage_infra": drainage_cv_summaries,
    } if RUN_FEATURE_ABLATIONS else None,
    "buffer_sensitivity": buffer_sensitivity_rows,
    "outputs": {
        "html": OUTPUT_HTML,
        "output_dir": OUTPUT_DIR,
        "metrics_json": METRICS_JSON,
        "cv_results_csv": CV_RESULTS_CSV,
        "feature_importance_csv": FEATURE_IMPORTANCE_CSV,
        "topk_summary_csv": TOPK_SUMMARY_CSV,
        "selected_fold_topk_csv": SELECTED_FOLD_TOPK_CSV,
        "risk_grade_summary_csv": RISK_GRADE_SUMMARY_CSV,
        "risk_grade_points_csv": RISK_GRADE_POINTS_CSV,
        "cumulative_risk_grade_points_csv": CUMULATIVE_RISK_GRADE_POINTS_CSV,
    },
}

save_experiment_outputs(
    metrics_payload,
    {
        CV_RESULTS_CSV: cv_result_rows,
        FEATURE_IMPORTANCE_CSV: importance_summary,
        TOPK_SUMMARY_CSV: topk_summary_rows,
        SELECTED_FOLD_TOPK_CSV: hybrid_result.get("hotspot_metrics", []),
        RISK_GRADE_SUMMARY_CSV: risk_grade_summaries,
        RISK_GRADE_POINTS_CSV: risk_grade_point_metrics,
        CUMULATIVE_RISK_GRADE_POINTS_CSV: cumulative_risk_grade_point_metrics,
    },
)

if GENERATE_MAP_OUTPUTS:
    # 결과 확인용 대화형 HTML 지도를 만든다.
    # 경계, 기준점, AlphaEarth 임베딩, Hybrid 확률, hotspot 레이어를 함께 저장한다.
    centroid = seoul.centroid().coordinates().getInfo()
    lon, lat = centroid[0], centroid[1]
    rgb_bands = valid_band_names[:3]

    Map = geemap.Map(center=[lat, lon], zoom=11, lite_mode=True)
    Map.addLayer(
        ee.Image().paint(seoul_fc, 1, 2),
        {"palette": ["cyan"]},
        "Seoul boundary",
    )
    if BOUNDARY_BUFFER_M > 0:
        analysis_region_fc = ee.FeatureCollection([ee.Feature(analysis_region)])
        Map.addLayer(
            ee.Image().paint(analysis_region_fc, 1, 2),
            {"palette": ["#ff7f00"]},
            f"Analysis boundary (+{BOUNDARY_BUFFER_M}m)",
        )
    Map.addLayer(
        positive_points,
        {"color": "#542788"},
        "Analysis flood positives",
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
        risk_grade,
        {
            "min": 1,
            "max": RISK_GRADE_COUNT,
            "palette": RISK_GRADE_PALETTE,
        },
        "Hybrid RF risk grade",
    )
    Map.addLayer(
        hotspots,
        {"palette": ["red"]},
        f"Hybrid hotspots (top {100 - HOTSPOT_PERCENTILE}%)",
    )
    Map.addLayerControl()
    Map.to_html(OUTPUT_HTML)
    sanitize_exported_widget_controls(OUTPUT_HTML)
    inject_static_legend(OUTPUT_HTML, "Hybrid RF risk grade", risk_grade_legend)
    print(f"Saved: {OUTPUT_HTML}")
else:
    print("HTML map generation skipped: GENERATE_MAP_OUTPUTS=0")
