"""One-time preprocessing: build data/mn_lakes.geojson (lake polygons keyed by DOW).

The interactive map (make_minnesota_map.py) draws each model's selected lakes as their
actual lake shapes when this file is present. The shapes come from the Minnesota DNR
Hydrography Dataset, whose open-water polygons carry the DOW lake number in the
``dowlknum`` field.

Get the source shapefile once from the Minnesota Geospatial Commons:
    https://gisdata.mn.gov/dataset/water-dnr-hydrography
Download the shapefile package and note the path to the open-water-polygon layer
(commonly ``dnr_hydro_features_all.shp``).

Then run:
    python scripts/prepare_lake_shapes.py \
        --hydro path/to/dnr_hydro_features_all.shp \
        --lake-info data/lake_info.csv \
        --out data/mn_lakes.geojson

Only lakes that appear in lake_info.csv (the study network) are kept, so the output is
small. Requires geopandas (needed only for this preprocessing step, not for the map).
"""

from __future__ import annotations

import argparse

import pandas as pd
import geopandas as gpd


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hydro", required=True,
                    help="DNR hydrography open-water polygon shapefile (has a dowlknum field)")
    ap.add_argument("--lake-info", default="data/lake_info.csv")
    ap.add_argument("--dow-field", default="dowlknum",
                    help="name of the DOW field in the hydrography layer")
    ap.add_argument("--simplify", type=float, default=0.0005,
                    help="geometry simplification tolerance in degrees (0 to disable)")
    ap.add_argument("--out", default="data/mn_lakes.geojson")
    args = ap.parse_args()

    study = set(pd.read_csv(args.lake_info)["DOW"].astype("int64"))

    g = gpd.read_file(args.hydro)
    if args.dow_field not in g.columns:
        raise SystemExit(f"field '{args.dow_field}' not in layer; columns: {list(g.columns)}")

    # normalize DOW to int and keep only the study lakes
    g["DOW"] = pd.to_numeric(g[args.dow_field], errors="coerce")
    g = g.dropna(subset=["DOW"])
    g["DOW"] = g["DOW"].astype("int64")
    g = g[g["DOW"].isin(study)]
    print(f"matched {g['DOW'].nunique()} of {len(study)} study lakes in the hydrography layer")

    # one geometry per DOW (a lake may be split into several polygons)
    g = g.dissolve(by="DOW", as_index=False)[["DOW", "geometry"]].to_crs("EPSG:4326")
    if args.simplify > 0:
        g["geometry"] = g["geometry"].simplify(args.simplify, preserve_topology=True)

    g.to_file(args.out, driver="GeoJSON")
    print(f"wrote {args.out} with {len(g)} lake polygons")


if __name__ == "__main__":
    main()
