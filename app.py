import streamlit as st
from streamlit_folium import st_folium
import folium
from folium.plugins import Draw
import asf_search as asf
from shapely.geometry import shape
import pandas as pd
from datetime import date, timedelta, datetime
import numpy as np
import cv2
import scipy.ndimage
import rasterio
from rasterio.vrt import WarpedVRT
import matplotlib.pyplot as plt
import io
import math
import os
import json
from shapely.geometry import mapping
from shapely import wkt as shapely_wkt
from shapely.ops import transform as shp_transform
from rasterio.mask import mask as rio_mask
from rasterio.warp import transform_geom
from pyproj import Transformer

def aoi_area_km2(wkt: str) -> float:
    geom = shapely_wkt.loads(wkt)
    lon, lat = geom.centroid.x, geom.centroid.y
    utm_zone = int((lon + 180) / 6) + 1
    epsg = 32600 + utm_zone if lat >= 0 else 32700 + utm_zone
    t = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
    return shp_transform(t.transform, geom).area / 1e6

st.set_page_config(page_title="SAR Flood Scene Selector", layout="wide")
st.title("SAR Flood Detection — Scene Selector")
st.caption(
    "Find and pair Sentinel-1 scenes for before/after flood comparison. "
    "Acquisition mode, orbit direction, and relative orbit are locked automatically "
    "for geometric consistency."
)


# ── helpers ───────────────────────────────────────────────────────────────────

def _read_and_clip(src, aoi_wkt):
    if aoi_wkt and src.crs is not None:
        geom_wgs84 = mapping(shapely_wkt.loads(aoi_wkt))
        geom_proj = transform_geom("EPSG:4326", src.crs, geom_wgs84)
        arr, clip_transform = rio_mask(src, [geom_proj], crop=True, nodata=0)
        arr = arr[0].astype(np.float32)
    else:
        arr = src.read(1).astype(np.float32)
        clip_transform = src.transform
    return arr, clip_transform


def load_sar_db(path, aoi_wkt=None) -> tuple[np.ndarray, dict]:
    with rasterio.open(path) as src:
        # Sentinel-1 GRD TIFFs store georeferencing as GCPs, not affine.
        # WarpedVRT converts them to a regular EPSG:4326 grid on the fly.
        with WarpedVRT(src) as vrt:
            arr, clip_transform = _read_and_clip(vrt, aoi_wkt)
            meta = vrt.meta.copy()
    meta.update({
        "count": 1,
        "dtype": "float32",
        "height": arr.shape[0],
        "width": arr.shape[1],
        "transform": clip_transform,
    })
    arr = np.clip(arr, 1e-6, None)
    arr_db = 10 * np.log10(arr)
    arr_db = scipy.ndimage.uniform_filter(arr_db, size=7)
    return arr_db, meta


def to_uint8(arr_db: np.ndarray) -> np.ndarray:
    return cv2.normalize(arr_db, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def register_sar(pre_db, post_db, nfeatures=5000, lowe_ratio=0.75, ransac_thresh=4.0):
    pre_u8, post_u8 = to_uint8(pre_db), to_uint8(post_db)
    sift = cv2.SIFT_create(nfeatures=nfeatures)
    kp1, des1 = sift.detectAndCompute(pre_u8, None)
    kp2, des2 = sift.detectAndCompute(post_u8, None)

    if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
        return pre_db, None, 0, 0.0

    matcher = cv2.BFMatcher(cv2.NORM_L2)
    pairs = matcher.knnMatch(des1, des2, k=2)
    good = [m for m, n in pairs if len((m, n)) == 2 and m.distance < lowe_ratio * n.distance]

    if len(good) < 4:
        return pre_db, None, len(good), 0.0

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransac_thresh)

    if H is None:
        return pre_db, None, len(good), 0.0

    inliers = int(mask.sum())
    ratio = inliers / len(good)
    h, w = post_db.shape
    pre_registered = cv2.warpPerspective(pre_db, H, (w, h))
    return pre_registered, H, inliers, ratio


def detect_flood(pre_db, post_db, threshold_db):
    diff = post_db - pre_db
    flood_mask = (diff < threshold_db).astype(np.uint8)
    return diff, flood_mask


def fig_to_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    return buf


def plot_sar(arr_db, title, vmin=None, vmax=None):
    # Skip nodata border pixels (zeros clip to ~-60 dB) before percentile calc
    valid = arr_db[np.isfinite(arr_db) & (arr_db > -55)]
    if valid.size == 0:
        valid = arr_db[np.isfinite(arr_db)]
    if vmin is None:
        vmin = float(np.percentile(valid, 5))
    if vmax is None:
        vmax = float(np.percentile(valid, 95))
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(arr_db, cmap="gray", vmin=vmin, vmax=vmax)
    ax.set_title(f"{title}\n(dB {vmin:.1f} – {vmax:.1f})", fontsize=9)
    ax.axis("off")
    return fig


def plot_flood_overlay(post_db, flood_mask):
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(post_db, cmap="gray", vmin=-25, vmax=0)
    overlay = np.zeros((*flood_mask.shape, 4), dtype=np.float32)
    overlay[flood_mask == 1] = [0, 0.5, 1, 0.6]   # blue = flood
    ax.imshow(overlay)
    ax.set_title("Flood mask (blue)")
    ax.axis("off")
    return fig


def aoi_wkt_from_drawing(drawing):
    if not drawing:
        return None
    return shape(drawing["geometry"]).wkt


def query_asf(wkt, start_date, end_date, beam_mode, direction, polarization):
    results = asf.search(
        platform=[asf.PLATFORM.SENTINEL1],
        intersectsWith=wkt,
        start=datetime.combine(start_date, datetime.min.time()),
        end=datetime.combine(end_date, datetime.max.time()),
        beamMode=[beam_mode],
        flightDirection=direction,
        polarization=[polarization],
        processingLevel=["GRD_HD", "GRD_MS"],  # Level-1 GRD only — RAW/SLC have no GeoTIFFs
        maxResults=100,
    )
    if not results:
        return pd.DataFrame()
    rows = []
    for r in results:
        p = r.properties
        rows.append({
            "Scene": p["sceneName"],
            "Date": p["startTime"][:10],
            "Rel. Orbit": p["pathNumber"],
            "Orbit Dir": p["flightDirection"],
            "Mode": p["beamModeType"],
            "Polarization": p["polarization"],
            "URL": p.get("url", ""),
        })
    return pd.DataFrame(rows).sort_values("Date", ascending=False).reset_index(drop=True)


# ── Step 1: AOI map ───────────────────────────────────────────────────────────

st.subheader("Step 1 — Draw your Area of Interest")

m = folium.Map(location=[23.7, 90.4], zoom_start=5, tiles="CartoDB positron")
Draw(
    draw_options={
        "rectangle": True, "polygon": False, "circle": False,
        "circlemarker": False, "polyline": False, "marker": False,
    },
    edit_options={"edit": False},
).add_to(m)
map_data = st_folium(m, width="100%", height=420, key="aoi_map")

aoi_wkt = aoi_wkt_from_drawing(
    map_data.get("last_active_drawing") if map_data else None
)

c1, c2 = st.columns(2)
with c1:
    if aoi_wkt:
        _area = aoi_area_km2(aoi_wkt)
        st.success(f"AOI captured — {_area:.1f} km²")
        st.download_button(
            "Save AOI to file",
            data=json.dumps({"wkt": aoi_wkt}),
            file_name="aoi.json",
            mime="application/json",
        )
    else:
        st.info("Draw a rectangle on the map to define your Area of Interest.")

with c2:
    uploaded_aoi = st.file_uploader("Load saved AOI (.json)", type=["json"], key="aoi_upload")
    if uploaded_aoi:
        aoi_wkt = json.load(uploaded_aoi).get("wkt")
        if aoi_wkt:
            _area = aoi_area_km2(aoi_wkt)
            st.success(f"AOI loaded — {_area:.1f} km²")

st.divider()

# ── Step 2: Filters ───────────────────────────────────────────────────────────

st.subheader("Step 2 — Filters")

c1, c2, c3, c4 = st.columns(4)
with c1:
    flood_date = st.date_input("Flood event date", value=date(2020, 7, 20))
with c2:
    beam_mode = st.selectbox("Acquisition mode", ["IW", "EW", "SM"])
with c3:
    direction = st.selectbox("Orbit direction", ["DESCENDING", "ASCENDING"])
with c4:
    polarization = st.selectbox("Polarization", ["VV+VH", "VV", "VH"])

pre_window = st.slider(
    "Pre-flood search window (days before event)", min_value=10, max_value=60, value=30,
    help="Sentinel-1 revisits the same orbit every ~6 days. 30 days gives ~5 candidate scenes."
)

st.caption(
    "Acquisition mode, orbit direction, and polarization apply to both searches. "
    "Relative orbit is locked automatically once you select your post-flood scene."
)

st.divider()

# ── Step 3: Post-flood scenes ─────────────────────────────────────────────────

st.subheader("Step 3 — Select post-flood scene")

if st.button(
    "Search post-flood scenes", type="primary", disabled=(aoi_wkt is None),
    help="Draw an AOI first." if aoi_wkt is None else None,
):
    with st.spinner("Querying Alaska Satellite Facility..."):
        df = query_asf(
            aoi_wkt,
            flood_date - timedelta(days=3),
            flood_date + timedelta(days=3),
            beam_mode, direction, polarization,
        )
        st.session_state["post_scenes"] = df
        # clear downstream state when re-searching
        for key in ("selected_post", "pre_scenes", "selected_pre"):
            st.session_state.pop(key, None)

if "post_scenes" in st.session_state:
    df_post = st.session_state["post_scenes"]
    if df_post.empty:
        st.warning("No post-flood scenes found. Try adjusting the date or filters.")
    else:
        st.dataframe(df_post.drop(columns=["URL"]), use_container_width=True)
        selected_name = st.selectbox(
            "Select post-flood scene", df_post["Scene"].tolist(), key="post_select"
        )
        scene_post = df_post[df_post["Scene"] == selected_name].iloc[0]
        st.session_state["selected_post"] = scene_post
        st.info(
            f"Relative orbit **{scene_post['Rel. Orbit']}** will be locked "
            "for the pre-flood search."
        )

st.divider()

# ── Step 4: Pre-flood scenes ──────────────────────────────────────────────────

if "selected_post" in st.session_state:
    locked_orbit = st.session_state["selected_post"]["Rel. Orbit"]

    st.subheader("Step 4 — Select pre-flood scene")
    st.caption(
        f"Relative orbit locked to **{locked_orbit}** (same as post-flood scene). "
        f"Searching {pre_window} days before the flood date."
    )

    if st.button("Search pre-flood scenes", type="primary"):
        with st.spinner("Querying Alaska Satellite Facility..."):
            pre_end = flood_date - timedelta(days=1)
            pre_start = pre_end - timedelta(days=pre_window)
            df = query_asf(aoi_wkt, pre_start, pre_end, beam_mode, direction, polarization)
            # lock relative orbit — this is the key consistency constraint
            df = df[df["Rel. Orbit"] == locked_orbit].reset_index(drop=True)
            st.session_state["pre_scenes"] = df
            st.session_state.pop("selected_pre", None)

    if "pre_scenes" in st.session_state:
        df_pre = st.session_state["pre_scenes"]
        if df_pre.empty:
            st.warning(
                f"No pre-flood scenes found for Relative Orbit {locked_orbit}. "
                "Sentinel-1 revisits every 6 days — try a longer search window."
            )
        else:
            st.dataframe(df_pre.drop(columns=["URL"]), use_container_width=True)
            selected_name = st.selectbox(
                "Select pre-flood scene", df_pre["Scene"].tolist(), key="pre_select"
            )
            scene_pre = df_pre[df_pre["Scene"] == selected_name].iloc[0]
            st.session_state["selected_pre"] = scene_pre
            st.success(f"Pre-flood scene selected: **{scene_pre['Date']}**")

    st.divider()

# ── Step 5: Summary + consistency check ──────────────────────────────────────

if "selected_post" in st.session_state and "selected_pre" in st.session_state:
    post = st.session_state["selected_post"]
    pre = st.session_state["selected_pre"]

    st.subheader("Step 5 — Scene pair summary")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Pre-flood date", pre["Date"])
        st.caption(pre["Scene"])
    with c2:
        st.metric("Post-flood date", post["Date"])
        st.caption(post["Scene"])
    with c3:
        delta = (
            datetime.strptime(post["Date"], "%Y-%m-%d")
            - datetime.strptime(pre["Date"], "%Y-%m-%d")
        ).days
        st.metric("Days apart", f"{delta} days")

    st.markdown("**Geometric consistency check:**")
    checks = {
        "Same relative orbit": pre["Rel. Orbit"] == post["Rel. Orbit"],
        "Same orbit direction": pre["Orbit Dir"] == post["Orbit Dir"],
        "Same acquisition mode": pre["Mode"] == post["Mode"],
        "Same polarization": pre["Polarization"] == post["Polarization"],
    }
    for label, passed in checks.items():
        st.write(f"{'✅' if passed else '❌'} {label}")

    st.divider()
    st.markdown("**Download instructions:**")
    st.markdown(
        "1. Create a free account at [urs.earthdata.nasa.gov](https://urs.earthdata.nasa.gov) if you don't have one\n"
        "2. Paste each URL below into your browser — you'll be prompted to log in, then the zip downloads automatically\n"
        "3. Unzip it — inside is a `.SAFE` folder with `.tiff` files in `measurement/`\n\n"
        "> **Note:** scenes are **GRD** (Level-1) products. "
        "If you downloaded a RAW product by mistake, delete it — RAW files are `.dat` and cannot be used here."
    )
    st.markdown(f"- **Pre-flood:** [{pre['Scene']}.zip]({pre['URL']})")
    st.markdown(f"- **Post-flood:** [{post['Scene']}.zip]({post['URL']})")

    # ── Step 6: Provide local file paths ─────────────────────────────────────

    st.subheader("Step 6 — Provide VV GeoTIFF paths")
    st.info(
        "**VV band only** — this pipeline uses only the VV (vertical–vertical) polarization channel. "
        "VV is the primary flood signal: flat water reflects radar energy away from the satellite, "
        "producing a sharp brightness drop compared to dry land. "
        "VH is sensitive to vegetation volume scattering and adds little extra for open-water detection."
    )
    st.markdown(
        "**How to find the VV file:**\n"
        "1. Unzip the downloaded `.zip` file\n"
        "2. Open the `.SAFE` folder → `measurement/`\n"
        "3. Find the file with `-vv-` in its name (e.g. `s1a-iw-grd-vv-....tiff`)\n"
        "4. Paste its full path below"
    )

    c1, c2 = st.columns(2)
    with c1:
        pre_path = st.text_input(
            "Pre-flood VV path",
            placeholder=r"C:\Downloads\S1A_...\measurement\s1a-iw-grd-vv-....tiff",
            key="pre_path",
            help="Full path to the VV polarization GeoTIFF inside the .SAFE/measurement/ folder.",
        )
        if pre_path and not os.path.isfile(pre_path):
            st.error("File not found.")
        elif pre_path:
            st.success("File found.")
    with c2:
        post_path = st.text_input(
            "Post-flood VV path",
            placeholder=r"C:\Downloads\S1A_...\measurement\s1a-iw-grd-vv-....tiff",
            key="post_path",
            help="Full path to the VV polarization GeoTIFF inside the .SAFE/measurement/ folder.",
        )
        if post_path and not os.path.isfile(post_path):
            st.error("File not found.")
        elif post_path:
            st.success("File found.")

    if pre_path and post_path and os.path.isfile(pre_path) and os.path.isfile(post_path):
        st.divider()

        # ── Step 7: Registration + flood detection ────────────────────────────

        st.subheader("Step 7 — Registration + Flood Detection")

        threshold_db = st.slider(
            "Flood threshold (dB drop)", min_value=-5.0, max_value=-0.5,
            value=-1.5, step=0.25,
            help=(
                "dB (decibel) = 10 × log₁₀(backscatter). A logarithmic scale that compresses "
                "the wide range of radar return values into a manageable range.\n\n"
                "Flood signal: water reflects radar away from the satellite → backscatter drops → "
                "post − pre VV is a large negative number.\n\n"
                "This threshold sets the minimum drop to call a pixel flooded. "
                "−1.5 dB ≈ 30% reduction in backscatter. "
                "Decrease (more negative) to reduce false alarms; increase toward 0 to catch subtle flooding."
            ),
        )

        with st.expander("Advanced registration settings", expanded=False):
            nfeatures = st.slider(
                "Max SIFT keypoints", 1000, 20000, 5000, 1000,
                help="Maximum keypoints SIFT detects per image. Increase if registration fails with too few inliers.",
            )
            lowe_ratio = st.slider(
                "Lowe's ratio", 0.5, 0.95, 0.75, 0.05,
                help="Match filter: keep a match only if its distance is < ratio × next-best distance. Lower = stricter.",
            )
            ransac_thresh = st.slider(
                "RANSAC threshold (px)", 1.0, 10.0, 4.0, 0.5,
                help="Max reprojection error (pixels) for a match to be counted as an inlier. Increase for large misalignments.",
            )

        if st.button("Run Analysis", type="primary"):
            with st.spinner("Loading and converting to dB..."):
                pre_db,  pre_meta  = load_sar_db(pre_path,  aoi_wkt)
                post_db, post_meta = load_sar_db(post_path, aoi_wkt)

            with st.spinner("Registering pre-flood image to post-flood..."):
                pre_registered, H, inliers, ratio = register_sar(
                    pre_db, post_db, nfeatures=nfeatures,
                    lowe_ratio=lowe_ratio, ransac_thresh=ransac_thresh,
                )

            with st.spinner("Detecting flood pixels..."):
                diff, flood_mask = detect_flood(pre_registered, post_db, threshold_db)

            # registration metrics
            flood_pct = 100 * flood_mask.sum() / flood_mask.size
            c1, c2, c3, c4, c5, c6 = st.columns(6)
            with c1:
                st.metric("RANSAC inliers", inliers,
                          help="Number of feature matches that fit the estimated homography within 4 px.")
            with c2:
                st.metric("Inlier ratio", f"{ratio:.2f}",
                          help="Fraction of matches kept after RANSAC. >0.5 is good; <0.2 suggests poor overlap.")
            with c3:
                st.metric("Flooded pixels", f"{flood_pct:.1f}%",
                          help="Percentage of AOI pixels flagged as flooded at the chosen threshold.")
            if H is not None:
                tx = H[0, 2]
                ty = H[1, 2]
                scale = math.sqrt(H[0, 0] ** 2 + H[1, 0] ** 2)
                angle = math.degrees(math.atan2(H[1, 0], H[0, 0]))
                with c4:
                    st.metric("Shift X", f"{tx:.1f} px",
                              help="Horizontal pixel offset applied to the pre-flood image to align it with post-flood.")
                with c5:
                    st.metric("Shift Y", f"{ty:.1f} px",
                              help="Vertical pixel offset. Near 0 for same-orbit scenes.")
                with c6:
                    st.metric("Rotation", f"{angle:.2f}°",
                              help="Rotation angle of the transform. Should be <1° for same-orbit SAR pairs; larger values indicate poor registration.")

            if H is None:
                st.warning("Registration failed — too few keypoints matched. Proceeding with unregistered images.")

            # store for saving
            st.session_state["results"] = {
                "pre_db": pre_db,
                "pre_registered": pre_registered,
                "post_db": post_db,
                "flood_mask": flood_mask,
                "meta": pre_meta,
            }

            # visualise
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                fig = plot_sar(pre_db, f"Pre-flood VV\n{pre['Date']}")
                st.pyplot(fig, use_container_width=True)
            with c2:
                fig = plot_sar(pre_registered, "Pre-flood (registered)")
                st.pyplot(fig, use_container_width=True)
            with c3:
                fig = plot_sar(post_db, f"Post-flood VV\n{post['Date']}")
                st.pyplot(fig, use_container_width=True)
            with c4:
                fig = plot_flood_overlay(post_db, flood_mask)
                st.pyplot(fig, use_container_width=True)

        if "results" in st.session_state:
            st.divider()
            st.subheader("Save outputs")
            default_out = os.path.dirname(pre_path) if pre_path else ""
            out_dir = st.text_input(
                "Output folder", value=default_out, key="out_dir",
                help=(
                    "Four GeoTIFFs will be saved here:\n"
                    "• pre_flood_vv_db.tif — pre-flood VV in dB (float32)\n"
                    "• pre_flood_registered_db.tif — pre-flood aligned to post (float32)\n"
                    "• post_flood_vv_db.tif — post-flood VV in dB (float32)\n"
                    "• flood_mask.tif — binary flood mask: 1=flood, 0=no flood (uint8)\n\n"
                    "dB files carry the AOI CRS and transform — open directly in QGIS or ArcGIS."
                ),
            )
            if st.button("Save GeoTIFFs", type="secondary") and out_dir:
                res = st.session_state["results"]
                meta = res["meta"].copy()
                os.makedirs(out_dir, exist_ok=True)

                def _write(arr, fname, dtype):
                    m = {
                        "driver": "GTiff",
                        "dtype": dtype,
                        "count": 1,
                        "height": arr.shape[0],
                        "width": arr.shape[1],
                        "crs": meta.get("crs"),
                        "transform": meta.get("transform"),
                    }
                    with rasterio.open(os.path.join(out_dir, fname), "w", **m) as dst:
                        dst.write(arr.astype(dtype), 1)

                _write(res["pre_db"],        "pre_flood_vv_db.tif",       "float32")
                _write(res["pre_registered"], "pre_flood_registered_db.tif", "float32")
                _write(res["post_db"],       "post_flood_vv_db.tif",      "float32")
                _write(res["flood_mask"],    "flood_mask.tif",            "uint8")
                st.success(f"Saved 4 GeoTIFFs to: {out_dir}")
