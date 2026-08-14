"""Interactive Minnesota lake-inspection map for the Nash-informed planning study.

Layers (all toggleable via the layer control):
  * County boundaries       - black outlines (data/mn_counties.geojson).
  * Infested lakes          - red dots, radius proportional to w_total (all infested lakes).
  * Uninfested lakes        - green dots, radius proportional to w_total; only lakes with
                              w_total >= --min-uninfested-w are drawn, to reduce clutter.
                              w_total(lake) = incoming + outgoing boat-movement weight (boats1.csv).
  * One layer per model      - the lakes that model inspects at the chosen budget. If a lake-
                              polygon file is supplied (--lakes-geojson, with a "DOW" property),
                              the actual lake shapes are drawn in a distinct color per model;
                              otherwise a small square marker is used. Models: NSG, RPNE, BSW, SW, BGC, GC.

A fixed dot legend is added at the bottom-left.

Usage
-----
    python scripts/make_minnesota_map.py \
        --lake-info data/lake_info.csv --boats data/boats1.csv \
        --counties data/mn_counties.geojson \
        --lakes-geojson data/mn_lakes.geojson \
        --selected nash_informed/ebmc/results/minnesota/all_selected_lakes_100%.pkl \
        --models NSG RPNE BSW SW BGC GC \
        --min-uninfested-w 2000 \
        --out mn_lakes_with_grouped_markers.html
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import folium
from pyproj import Transformer

INFESTED_COLS = ["zm2019", "ss2019", "ew2019", "sf2019"]

COLOR_INFESTED = "#d62728"   # red
COLOR_CLEAN = "#2ca02c"      # green

MODEL_COLORS = {
    "NSG":  "#7f7f7f", "RPNE": "#ff7f0e", "BSW": "#1f77b4",
    "SW":   "#9467bd", "BGC":  "#8c564b", "GC":  "#17becf",
}

R_MIN, R_MAX = 2.0, 16.0      # dot-radius range (pixels), area ~ proportional to w_total
POLY_HALF = 0.018             # half-side (deg) of the square fallback marker


def load_lakes(lake_info_csv: str, boats_csv: str) -> pd.DataFrame:
    df = pd.read_csv(lake_info_csv)
    tr = Transformer.from_crs("EPSG:26915", "EPSG:4326", always_xy=True)
    lon, lat = tr.transform(df["utm_x"].values, df["utm_y"].values)
    df["lat"], df["lon"] = lat, lon
    df["infested"] = df[INFESTED_COLS].fillna(0).astype(int).sum(axis=1) > 0
    boats = pd.read_csv(boats_csv)
    w_total = (boats.groupby("dow_origin")["weight"].sum()
               .add(boats.groupby("dow_destination")["weight"].sum(), fill_value=0))
    df["w_total"] = df["DOW"].map(w_total).fillna(0.0)
    return df


def _radius(w: float, w_max: float) -> float:
    return R_MIN if w_max <= 0 else R_MIN + (R_MAX - R_MIN) * np.sqrt(max(w, 0.0) / w_max)


def _square(lat: float, lon: float, half: float = POLY_HALF):
    return [[lat - half, lon - half], [lat - half, lon + half],
            [lat + half, lon + half], [lat + half, lon - half]]


def add_county_layer(m: folium.Map, path: str) -> None:
    folium.GeoJson(path, name="County boundaries",
                   style_function=lambda _f: {"color": "black", "weight": 1.2,
                                              "fill": False, "opacity": 0.8}).add_to(m)


def add_lake_dot_layers(m: folium.Map, df: pd.DataFrame, min_uninfested_w: float) -> None:
    w_max = df["w_total"].max()
    specs = [
        ("Infested lakes (size ∝ w_total)", df[df["infested"]], COLOR_INFESTED, True),
        (f"Uninfested lakes (w_total ≥ {min_uninfested_w:g})",
         df[(~df["infested"]) & (df["w_total"] >= min_uninfested_w)], COLOR_CLEAN, False),
    ]
    for label, sub, color, show in specs:
        group = folium.FeatureGroup(name=label, show=show)
        for _, r in sub.iterrows():
            folium.CircleMarker(
                location=[r["lat"], r["lon"]], radius=_radius(r["w_total"], w_max),
                color=color, fill=True, fill_color=color, fill_opacity=0.7, weight=0,
                popup=folium.Popup(
                    f"<b>{r.get('lake_name','')}</b><br>DOW {r['DOW']}<br>"
                    f"{r.get('county_name','')}<br>w_total = {r['w_total']:.0f}", max_width=250),
            ).add_to(group)
        group.add_to(m)


def _load_lake_shapes(path: str) -> dict:
    """Return {DOW -> geometry (GeoJSON geometry dict)} from a lake-polygon GeoJSON."""
    with open(path) as f:
        fc = json.load(f)
    shapes = {}
    for feat in fc.get("features", []):
        dow = feat.get("properties", {}).get("DOW")
        if dow is not None:
            shapes[int(dow)] = feat["geometry"]
    return shapes


def add_model_layers(m: folium.Map, df: pd.DataFrame, selected: dict,
                     models: list[str], lake_shapes: dict | None) -> None:
    rows = df.set_index("DOW")
    for model in models:
        if model not in selected:
            print(f"  [skip] {model} not in selection file")
            continue
        color = MODEL_COLORS.get(model, "#333333")
        sel = [int(x) for x in selected[model]]
        lakes = rows.loc[rows.index.intersection(sel)]
        n_inf = int(lakes["infested"].sum())
        group = folium.FeatureGroup(
            name=f"{model} plan — {len(lakes)} lakes ({n_inf} infested)", show=(model == "BSW"))
        for dow, r in lakes.iterrows():
            popup = f"{model}: DOW {int(dow)} ({r.get('county_name','')})"
            if lake_shapes and int(dow) in lake_shapes:
                folium.GeoJson(
                    {"type": "Feature", "geometry": lake_shapes[int(dow)], "properties": {}},
                    style_function=lambda _f, c=color: {"color": c, "weight": 1,
                                                        "fill": True, "fillColor": c,
                                                        "fillOpacity": 0.55},
                    popup=popup).add_to(group)
            else:
                folium.Polygon(locations=_square(r["lat"], r["lon"]), color=color, weight=1,
                               fill=True, fill_color=color, fill_opacity=0.45,
                               popup=popup).add_to(group)
        group.add_to(m)


def add_legend(m: folium.Map, min_uninfested_w: float) -> None:
    html = f"""
    <div style="position: fixed; bottom: 22px; left: 22px; z-index: 9999; background: white;
        padding: 10px 12px; border: 1px solid #888; border-radius: 6px; font-size: 13px;
        line-height: 1.5; box-shadow: 0 1px 4px rgba(0,0,0,0.3);">
      <b>Dot legend</b><br>
      <span style="color:{COLOR_INFESTED}; font-size:16px;">&#9679;</span> Infested lake<br>
      <span style="color:{COLOR_CLEAN}; font-size:16px;">&#9679;</span> Uninfested lake
      (w<sub>total</sub> &ge; {min_uninfested_w:g})<br>
      <span style="font-size:11px; color:#555;">dot size &#8733; w<sub>total</sub>
      (incoming + outgoing boat traffic)</span>
    </div>"""
    m.get_root().html.add_child(folium.Element(html))


def build_map(df, selected, models, counties, lakes_geojson, min_uninfested_w) -> folium.Map:
    m = folium.Map(location=[46.3, -94.3], zoom_start=6, tiles="CartoDB positron",
                   control_scale=True)
    if counties and Path(counties).exists():
        add_county_layer(m, counties)
    add_lake_dot_layers(m, df, min_uninfested_w)
    shapes = _load_lake_shapes(lakes_geojson) if (lakes_geojson and Path(lakes_geojson).exists()) else None
    if lakes_geojson and shapes is None:
        print(f"  [note] {lakes_geojson} not found; drawing model lakes as square markers.")
    add_model_layers(m, df, selected, models, shapes)
    folium.LayerControl(collapsed=False).add_to(m)
    add_legend(m, min_uninfested_w)
    return m


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lake-info", default="data/lake_info.csv")
    ap.add_argument("--boats", default="data/boats1.csv")
    ap.add_argument("--counties", default="data/mn_counties.geojson")
    ap.add_argument("--lakes-geojson", default="data/mn_lakes.geojson",
                    help="lake-polygon GeoJSON with a DOW property; falls back to squares if absent")
    ap.add_argument("--selected",
                    default="nash_informed/ebmc/results/minnesota/all_selected_lakes_100%.pkl")
    ap.add_argument("--models", nargs="+", default=["NSG", "RPNE", "BSW", "SW", "BGC", "GC"])
    ap.add_argument("--min-uninfested-w", type=float, default=2000.0)
    ap.add_argument("--out", default="mn_lakes_with_grouped_markers.html")
    args = ap.parse_args()

    df = load_lakes(args.lake_info, args.boats)
    with open(args.selected, "rb") as f:
        selected = pickle.load(f)
    print(f"loaded {len(df)} lakes ({int(df['infested'].sum())} infested); "
          f"w_total {df['w_total'].min():.0f}-{df['w_total'].max():.0f}; keys: {list(selected)}")
    m = build_map(df, selected, args.models, args.counties, args.lakes_geojson,
                  args.min_uninfested_w)
    m.save(args.out)
    print(f"wrote {args.out}  ({Path(args.out).stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
