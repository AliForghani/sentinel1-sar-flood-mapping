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
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.features import geometry_mask
from rasterio.transform import from_origin
from rasterio.vrt import WarpedVRT
from skimage.registration import phase_cross_correlation
import matplotlib.pyplot as plt
import io
import math
import os
import json
from shapely.geometry import mapping
from shapely import wkt as shapely_wkt
from shapely.ops import transform as shp_transform
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

def aoi_grid(post_path, aoi_wkt) -> dict:
    """Common EPSG:4326 pixel grid covering the AOI, at the post-flood scene's native resolution.

    Both scenes are warped onto this exact grid, so pixel (i, j) is the same ground
    location in both — no image-to-image warping is needed afterwards.
    """
    with rasterio.open(post_path) as src, WarpedVRT(src) as vrt:
        res_x, res_y = vrt.res
    minx, miny, maxx, maxy = shapely_wkt.loads(aoi_wkt).bounds
    width = max(1, math.ceil((maxx - minx) / res_x))
    height = max(1, math.ceil((maxy - miny) / res_y))
    return {
        "crs": CRS.from_epsg(4326),
        "transform": from_origin(minx, maxy, res_x, res_y),
        "width": width,
        "height": height,
    }


def nan_uniform_filter(arr, size=7):
    """Mean filter that ignores NaN (nodata) pixels instead of smearing them into valid ones."""
    valid = np.isfinite(arr)
    num = scipy.ndimage.uniform_filter(np.where(valid, arr, 0).astype(np.float32), size=size)
    den = scipy.ndimage.uniform_filter(valid.astype(np.float32), size=size)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = num / den
    out[~valid] = np.nan
    return out


def load_sar_db(path, grid, aoi_wkt) -> np.ndarray:
    """Read a GRD VV TIFF onto `grid` and return dB values, NaN where there is no data."""
    with rasterio.open(path) as src:
        # Sentinel-1 GRD TIFFs store georeferencing as GCPs, not affine.
        # WarpedVRT fits the GCPs and resamples directly onto the shared AOI grid.
        with WarpedVRT(
            src,
            crs=grid["crs"], transform=grid["transform"],
            width=grid["width"], height=grid["height"],
            src_nodata=0, nodata=0,  # GRD border pixels are 0
            resampling=Resampling.bilinear,
        ) as vrt:
            dn = vrt.read(1).astype(np.float32)
    inside_aoi = geometry_mask(
        [mapping(shapely_wkt.loads(aoi_wkt))],
        out_shape=dn.shape, transform=grid["transform"], invert=True,
    )
    valid = (dn > 0) & inside_aoi
    arr_db = np.full(dn.shape, np.nan, dtype=np.float32)
    arr_db[valid] = 10 * np.log10(dn[valid])
    return nan_uniform_filter(arr_db, size=7)


def register_phase(pre_db, post_db, max_shift_px=3.0, upsample_factor=10):
    """Estimate the residual sub-pixel shift between two images on the same grid.

    Uses phase correlation (translation only). The shift is applied only if it is
    smaller than `max_shift_px`; a larger value means the estimate is unreliable
    (or the georeferencing itself is wrong), so the pre-flood image is left as is.

    Returns (pre_registered, info).
    """
    info = {"dx": 0.0, "dy": 0.0, "error": float("nan"), "applied": False, "reason": ""}
    common = np.isfinite(pre_db) & np.isfinite(post_db)
    if common.sum() < 1000:
        info["reason"] = "too little overlap between the two images"
        return pre_db, info

    # Phase correlation needs finite input: fill nodata with each image's mean.
    pre_f = np.where(np.isfinite(pre_db), pre_db, np.nanmean(pre_db[common]))
    post_f = np.where(np.isfinite(post_db), post_db, np.nanmean(post_db[common]))

    shift, error, _ = phase_cross_correlation(
        post_f, pre_f, upsample_factor=upsample_factor
    )
    info.update(dy=float(shift[0]), dx=float(shift[1]), error=float(error))

    if math.hypot(info["dx"], info["dy"]) > max_shift_px:
        info["reason"] = f"estimated shift exceeds {max_shift_px} px"
        return pre_db, info

    # cval=NaN: pixels shifted in from outside the image are marked as nodata.
    pre_registered = scipy.ndimage.shift(pre_db, (info["dy"], info["dx"]), order=1, cval=np.nan)
    info["applied"] = True
    return pre_registered.astype(np.float32), info


def to_uint8(arr_db: np.ndarray) -> np.ndarray:
    """Stretch valid dB values (2nd–98th percentile) to 0–255 for SIFT; nodata becomes 0."""
    valid = np.isfinite(arr_db)
    lo, hi = np.percentile(arr_db[valid], [2, 98])
    out = np.zeros(arr_db.shape, dtype=np.uint8)
    out[valid] = (np.clip((arr_db[valid] - lo) / (hi - lo + 1e-6), 0, 1) * 255).astype(np.uint8)
    return out


def register_sift(pre_db, post_db, nfeatures=5000, lowe_ratio=0.75, ransac_thresh=4.0,
                  min_inliers=20, max_shift_px=3.0, max_rotation_deg=1.0):
    """Align pre-flood to post-flood with SIFT keypoints + RANSAC homography.

    The homography is applied only if it passes sanity checks (enough inliers,
    small shift, small rotation); otherwise the pre-flood image is left as is.

    Returns (pre_registered, info).
    """
    info = {"dx": 0.0, "dy": 0.0, "matches": 0, "inliers": 0, "inlier_ratio": 0.0,
            "rotation": 0.0, "applied": False, "reason": ""}
    pre_valid, post_valid = np.isfinite(pre_db), np.isfinite(post_db)
    if pre_valid.sum() < 1000 or post_valid.sum() < 1000:
        info["reason"] = "too few valid pixels"
        return pre_db, info

    # Only detect keypoints well inside valid data — the edge between image and
    # nodata is a strong artificial feature that would otherwise dominate matching.
    kernel = np.ones((15, 15), np.uint8)
    pre_kp_mask = cv2.erode(pre_valid.astype(np.uint8) * 255, kernel)
    post_kp_mask = cv2.erode(post_valid.astype(np.uint8) * 255, kernel)

    sift = cv2.SIFT_create(nfeatures=nfeatures)
    kp1, des1 = sift.detectAndCompute(to_uint8(pre_db), pre_kp_mask)
    kp2, des2 = sift.detectAndCompute(to_uint8(post_db), post_kp_mask)
    if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
        info["reason"] = "too few keypoints detected"
        return pre_db, info

    # For each pre-flood keypoint, find the 2 nearest post-flood keypoints,
    # then keep it only if the best is clearly better than the second (Lowe's ratio test).
    pairs = cv2.BFMatcher(cv2.NORM_L2).knnMatch(des1, des2, k=2)
    good = [p[0] for p in pairs if len(p) == 2 and p[0].distance < lowe_ratio * p[1].distance]
    info["matches"] = len(good)
    if len(good) < 4:
        info["reason"] = "too few matches after Lowe's ratio test"
        return pre_db, info

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    H, inlier_mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransac_thresh)
    if H is None:
        info["reason"] = "RANSAC could not fit a homography"
        return pre_db, info

    info["inliers"] = int(inlier_mask.sum())
    info["inlier_ratio"] = info["inliers"] / len(good)
    info["dx"], info["dy"] = float(H[0, 2]), float(H[1, 2])
    info["rotation"] = math.degrees(math.atan2(H[1, 0], H[0, 0]))

    if info["inliers"] < min_inliers:
        info["reason"] = f"only {info['inliers']} inliers (need ≥ {min_inliers})"
    elif math.hypot(info["dx"], info["dy"]) > max_shift_px:
        info["reason"] = f"estimated shift exceeds {max_shift_px} px"
    elif abs(info["rotation"]) > max_rotation_deg:
        info["reason"] = f"estimated rotation exceeds {max_rotation_deg}°"
    if info["reason"]:
        return pre_db, info

    # borderValue=NaN: areas warped in from outside the image are marked as nodata.
    h, w = post_db.shape
    pre_registered = cv2.warpPerspective(
        pre_db.astype(np.float32), H, (w, h),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=float("nan"),
    )
    info["applied"] = True
    return pre_registered, info


def detect_flood(pre_db, post_db, threshold_db):
    """Return the dB difference and a mask: 1 = flood, 0 = no flood, 255 = nodata."""
    diff = post_db - pre_db
    valid = np.isfinite(diff)
    flood_mask = np.full(diff.shape, 255, dtype=np.uint8)
    flood_mask[valid] = (diff[valid] < threshold_db).astype(np.uint8)
    return diff, flood_mask


def fig_to_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    return buf


def plot_sar(arr_db, title, vmin=None, vmax=None):
    # Nodata is NaN, so percentiles are computed over valid pixels only
    valid = arr_db[np.isfinite(arr_db)]
    if valid.size == 0:
        valid = np.array([0.0])
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
    valid = post_db[np.isfinite(post_db)]
    vmin, vmax = (np.percentile(valid, [5, 95]) if valid.size else (None, None))
    ax.imshow(post_db, cmap="gray", vmin=vmin, vmax=vmax)
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

        reg_method = st.radio(
            "Fine registration method (Stage 2)",
            ["SIFT + RANSAC homography", "Phase correlation (shift only)"],
            horizontal=True,
            help=(
                "SIFT + RANSAC: automatic feature matching — detects keypoints in both images, "
                "matches them, and fits a homography.\n\n"
                "Phase correlation: estimates a sub-pixel shift from the whole image. "
                "Simpler and more stable baseline — useful to compare against SIFT."
            ),
        )

        with st.expander("Advanced registration settings", expanded=False):
            max_shift_px = st.slider(
                "Max accepted shift (px)", 0.5, 10.0, 3.0, 0.5,
                help=(
                    "Both scenes are already on the same pixel grid, so the remaining offset should be small. "
                    "A larger estimated shift is treated as unreliable and not applied."
                ),
            )
            if reg_method.startswith("SIFT"):
                nfeatures = st.slider(
                    "Max SIFT keypoints", 1000, 20000, 5000, 1000,
                    help="Maximum keypoints SIFT detects per image. Increase if too few inliers are found.",
                )
                lowe_ratio = st.slider(
                    "Lowe's ratio", 0.5, 0.95, 0.75, 0.05,
                    help="Keep a match only if its distance is < ratio × next-best distance. Lower = stricter.",
                )
                ransac_thresh = st.slider(
                    "RANSAC threshold (px)", 1.0, 10.0, 4.0, 0.5,
                    help="Max reprojection error (pixels) for a match to count as an inlier.",
                )
                min_inliers = st.slider(
                    "Min inliers to accept", 4, 100, 20, 1,
                    help="The homography is applied only if at least this many matches agree with it.",
                )

        if st.button("Run Analysis", type="primary"):
            with st.spinner("Warping both scenes onto a common AOI grid and converting to dB..."):
                grid = aoi_grid(post_path, aoi_wkt)
                pre_db = load_sar_db(pre_path, grid, aoi_wkt)
                post_db = load_sar_db(post_path, grid, aoi_wkt)

            if reg_method.startswith("SIFT"):
                with st.spinner("Registering with SIFT + RANSAC..."):
                    pre_registered, reg = register_sift(
                        pre_db, post_db, nfeatures=nfeatures, lowe_ratio=lowe_ratio,
                        ransac_thresh=ransac_thresh, min_inliers=min_inliers,
                        max_shift_px=max_shift_px,
                    )
            else:
                with st.spinner("Estimating residual shift (phase correlation)..."):
                    pre_registered, reg = register_phase(
                        pre_db, post_db, max_shift_px=max_shift_px,
                    )

            with st.spinner("Detecting flood pixels..."):
                diff, flood_mask = detect_flood(pre_registered, post_db, threshold_db)

            # registration metrics
            n_valid = int((flood_mask != 255).sum())
            flood_pct = 100 * (flood_mask == 1).sum() / n_valid if n_valid else 0.0
            metrics = [
                ("Shift X", f"{reg['dx']:.2f} px",
                 "Estimated horizontal offset of the pre-flood image relative to post-flood."),
                ("Shift Y", f"{reg['dy']:.2f} px",
                 "Estimated vertical offset of the pre-flood image relative to post-flood."),
            ]
            if reg_method.startswith("SIFT"):
                metrics += [
                    ("Matches", reg["matches"], "Keypoint matches that passed Lowe's ratio test."),
                    ("RANSAC inliers", reg["inliers"],
                     "Matches consistent with the fitted homography."),
                    ("Inlier ratio", f"{reg['inlier_ratio']:.2f}",
                     "Fraction of matches kept by RANSAC. Low values mean many wrong matches."),
                    ("Rotation", f"{reg['rotation']:.2f}°",
                     "Rotation in the homography. Should be ~0° for same-orbit pairs."),
                ]
            else:
                metrics += [
                    ("Correlation error", f"{reg['error']:.2f}",
                     "Phase-correlation error (0 = perfect match, 1 = no match). Flooding itself raises it."),
                ]
            metrics += [
                ("Applied", "Yes" if reg["applied"] else "No",
                 "Whether the estimated transform passed the sanity checks and was applied."),
                ("Flooded pixels", f"{flood_pct:.1f}%",
                 "Percentage of valid AOI pixels flagged as flooded at the chosen threshold."),
            ]
            for col, (label, value, help_text) in zip(st.columns(len(metrics)), metrics):
                with col:
                    st.metric(label, value, help=help_text)

            if not reg["applied"]:
                st.warning(
                    f"Registration not applied — {reg['reason']}. "
                    "Using the georeferenced images as is (both are already on the same grid)."
                )

            # store for saving
            st.session_state["results"] = {
                "pre_db": pre_db,
                "pre_registered": pre_registered,
                "post_db": post_db,
                "flood_mask": flood_mask,
                "grid": grid,
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
                    "• flood_mask.tif — flood mask: 1=flood, 0=no flood, 255=nodata (uint8)\n\n"
                    "All four files share the same AOI grid (EPSG:4326) — open directly in QGIS or ArcGIS."
                ),
            )
            if st.button("Save GeoTIFFs", type="secondary") and out_dir:
                res = st.session_state["results"]
                grid = res["grid"]
                os.makedirs(out_dir, exist_ok=True)

                def _write(arr, fname, dtype, nodata):
                    m = {
                        "driver": "GTiff",
                        "dtype": dtype,
                        "count": 1,
                        "height": grid["height"],
                        "width": grid["width"],
                        "crs": grid["crs"],
                        "transform": grid["transform"],
                        "nodata": nodata,
                        "compress": "deflate",
                    }
                    with rasterio.open(os.path.join(out_dir, fname), "w", **m) as dst:
                        dst.write(arr.astype(dtype), 1)

                _write(res["pre_db"],         "pre_flood_vv_db.tif",         "float32", np.nan)
                _write(res["pre_registered"], "pre_flood_registered_db.tif", "float32", np.nan)
                _write(res["post_db"],        "post_flood_vv_db.tif",        "float32", np.nan)
                _write(res["flood_mask"],     "flood_mask.tif",              "uint8",   255)
                st.success(f"Saved 4 GeoTIFFs to: {out_dir}")
