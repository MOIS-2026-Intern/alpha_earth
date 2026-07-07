import html
import json
import os
import re

import ee
import geemap

run_analysis = None


def configure(run_analysis_module):
    global run_analysis
    run_analysis = run_analysis_module


def get_run_analysis():
    global run_analysis
    if run_analysis is None:
        try:
            from . import run_analysis as run_analysis_module
        except ImportError:
            import run_analysis as run_analysis_module
        configure(run_analysis_module)
    return run_analysis


def parse_int_list(raw_value):
    return [
        int(value.strip())
        for value in raw_value.split(",")
        if value.strip()
    ]


def output_html_path():
    analysis = get_run_analysis()
    return analysis.resolve_output_path(
        os.environ.get("FINAL_MAP_OUTPUT_HTML", "seoul_flood_risk_final.html")
    )


def top5_ranking_csv_path():
    analysis = get_run_analysis()
    return os.environ.get(
        "TOP5_RED_POINTS_BY_GU_CSV",
        os.path.join(analysis.OUTPUT_DIR, "top5_red_points_by_gu.csv"),
    )


HOTSPOT_PERCENTILE = int(os.environ.get("HOTSPOT_PERCENTILE", "95"))
RISK_GRADE_PERCENTILES = parse_int_list(
    os.environ.get("RISK_GRADE_PERCENTILES", "50,75,90,95")
)
RISK_GRADE_PALETTE = [
    value.strip()
    for value in os.environ.get(
        "RISK_GRADE_PALETTE",
        "#2b83ba,#abdda4,#ffffbf,#fdae61,#d7191c",
    ).split(",")
    if value.strip()
]
RISK_GRADE_NAMES = [
    value.strip()
    for value in os.environ.get(
        "RISK_GRADE_NAMES",
        "Very low,Low,Moderate,High,Very high",
    ).split(",")
    if value.strip()
]
RISK_GRADE_COUNT = len(RISK_GRADE_PERCENTILES) + 1

if (
    RISK_GRADE_PERCENTILES != sorted(RISK_GRADE_PERCENTILES)
    or not all(0 < percentile < 100 for percentile in RISK_GRADE_PERCENTILES)
):
    raise ValueError("RISK_GRADE_PERCENTILES must be ascending integers between 0 and 100.")
if len(RISK_GRADE_PALETTE) != RISK_GRADE_COUNT:
    raise ValueError("RISK_GRADE_PALETTE must match the number of risk grades.")
if len(RISK_GRADE_NAMES) != RISK_GRADE_COUNT:
    raise ValueError("RISK_GRADE_NAMES must match the number of risk grades.")


def build_risk_grade(probability):
    minmax_info = probability.reduceRegion(
        reducer=ee.Reducer.minMax(),
        geometry=run_analysis.seoul,
        scale=run_analysis.ANALYSIS_SCALE,
        maxPixels=1e9,
    ).getInfo()
    threshold_info = probability.reduceRegion(
        reducer=ee.Reducer.percentile(RISK_GRADE_PERCENTILES),
        geometry=run_analysis.seoul,
        scale=run_analysis.ANALYSIS_SCALE,
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
            geometry=run_analysis.seoul,
            scale=run_analysis.ANALYSIS_SCALE,
            maxPixels=1e9,
        )
        .getInfo()
    )
    area_by_grade = {
        int(group["grade"]): group["sum"]
        for group in area_info.get("groups", [])
    }

    grade_summaries = []
    lower_percentile = 0
    for grade in range(1, RISK_GRADE_COUNT + 1):
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


def build_risk_grade_legend(grade_summaries):
    legend = {}
    for row in grade_summaries:
        grade = row["grade"]
        label = f"Grade {grade}: {RISK_GRADE_NAMES[grade - 1]} ({row['percentile_range']})"
        legend[label] = RISK_GRADE_PALETTE[grade - 1]
    return legend


def inject_static_legend(html_path, title, legend_dict):
    items_html = "\n".join(
        (
            '<div class="risk-grade-legend-item">'
            f'<span class="risk-grade-legend-swatch" style="background:{html.escape(color, quote=True)}"></span>'
            f"<span>{html.escape(label)}</span>"
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
  max-width: 270px;
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


def percentile_threshold(probability, percentile):
    threshold_info = probability.reduceRegion(
        reducer=ee.Reducer.percentile([percentile]),
        geometry=run_analysis.seoul,
        scale=run_analysis.ANALYSIS_SCALE,
        maxPixels=1e9,
    ).getInfo()
    return threshold_info[f"flood_prob_p{percentile}"]


def seoul_gu_feature_collection():
    shape_to_gu = {
        shape_name: gu_name
        for gu_name, shape_name in run_analysis.SEOUL_GU_TO_ADM2_SHAPE.items()
    }
    shape_names = list(shape_to_gu)
    shape_to_gu_dict = ee.Dictionary(shape_to_gu)
    adm2 = (
        ee.FeatureCollection("WM/geoLab/geoBoundaries/600/ADM2")
        .filter(ee.Filter.eq("shapeGroup", "KOR"))
        .filterBounds(run_analysis.seoul)
        .filter(ee.Filter.inList("shapeName", shape_names))
    )

    def _set_gu_name(feature):
        return feature.set("gu_name", shape_to_gu_dict.get(feature.get("shapeName")))

    return adm2.map(_set_gu_name)


def normalize_stat_value(props, key):
    value = props.get(key)
    if value is None:
        value = props.get(f"{key}_sum")
    return value or 0


def compute_top_hotspot_rankings(hotspots):
    red_point_stats = (
        ee.Image.constant(1)
        .rename("red_point_count")
        .addBands(ee.Image.pixelArea().divide(1e6).rename("red_area_km2"))
        .updateMask(hotspots)
    )
    stats = red_point_stats.reduceRegions(
        collection=seoul_gu_feature_collection(),
        reducer=ee.Reducer.sum(),
        scale=run_analysis.ANALYSIS_SCALE,
        tileScale=4,
    ).getInfo()["features"]

    rows = []
    for feature in stats:
        props = feature["properties"]
        count = int(round(float(normalize_stat_value(props, "red_point_count"))))
        area_km2 = float(normalize_stat_value(props, "red_area_km2"))
        rows.append(
            {
                "gu_name": props.get("gu_name"),
                "adm2_shape_name": props.get("shapeName"),
                "red_point_count": count,
                "red_area_km2": area_km2,
            }
        )

    rows.sort(key=lambda row: (-row["red_point_count"], row["gu_name"] or ""))
    for index, row in enumerate(rows, start=1):
        row["rank"] = index
    return rows


def save_map(probability, risk_grade, hotspots, risk_grade_legend):
    centroid = run_analysis.seoul.centroid().coordinates().getInfo()
    lon, lat = centroid[0], centroid[1]
    output_html = output_html_path()

    flood_map = geemap.Map(center=[lat, lon], zoom=11, lite_mode=True)
    flood_map.addLayer(
        ee.Image().paint(run_analysis.seoul_fc, 1, 2),
        {"palette": ["cyan"]},
        "Seoul boundary",
    )
    flood_map.addLayer(
        run_analysis.positive_points,
        {"color": "#542788"},
        "Flood reference points",
    )
    flood_map.addLayer(
        probability,
        {"min": 0, "max": 1, "palette": ["#f7fbff", "#6baed6", "#2171b5", "#08306b"]},
        "Final GTB flood probability",
    )
    flood_map.addLayer(
        risk_grade,
        {
            "min": 1,
            "max": RISK_GRADE_COUNT,
            "palette": RISK_GRADE_PALETTE,
        },
        "Final GTB risk grade",
    )
    flood_map.addLayer(
        hotspots,
        {"palette": ["red"]},
        f"Top {100 - HOTSPOT_PERCENTILE}% high-risk points",
    )
    flood_map.addLayerControl()
    flood_map.to_html(output_html)
    sanitize_exported_widget_controls(output_html)
    inject_static_legend(output_html, "Final GTB risk grade", risk_grade_legend)
    return output_html


def create_map_outputs(analysis_result, run_analysis_module=None):
    if run_analysis_module is not None:
        configure(run_analysis_module)
    final_model = analysis_result["final_model"]

    classifier = final_model["classifier"]
    feature_image = final_model["feature_image"]
    probability = (
        run_analysis.classify_image_with_voting(
            feature_image,
            run_analysis.INPUT_BANDS,
            classifier,
            "flood_prob",
        )
        .clip(run_analysis.seoul)
    )

    risk_grade, thresholds, grade_summaries = build_risk_grade(probability)
    hotspot_threshold = thresholds.get(HOTSPOT_PERCENTILE)
    if hotspot_threshold is None:
        hotspot_threshold = percentile_threshold(probability, HOTSPOT_PERCENTILE)
    hotspots = probability.gte(ee.Number(hotspot_threshold)).selfMask().rename("hotspots")

    ranking_rows = compute_top_hotspot_rankings(hotspots)
    ranking_csv = top5_ranking_csv_path()
    run_analysis.write_csv(ranking_csv, ranking_rows)

    print(f"\nTop {100 - HOTSPOT_PERCENTILE}% red hotspot ranking by Seoul district:")
    for row in ranking_rows:
        print(
            f"{row['rank']:2d}. {row['gu_name']}: "
            f"{row['red_point_count']} red points, "
            f"{row['red_area_km2']:.3f} km2"
        )

    output_html = save_map(
        probability,
        risk_grade,
        hotspots,
        build_risk_grade_legend(grade_summaries),
    )
    print(f"Saved map: {output_html}")
    print(f"Saved ranking CSV: {ranking_csv}")
    return {
        "final_map_html": output_html,
        "top5_red_points_by_gu_csv": ranking_csv,
    }


def main():
    analysis = get_run_analysis()
    analysis_result = analysis.main(generate_final_map=False)
    return create_map_outputs(analysis_result, analysis)


if __name__ == "__main__":
    main()
