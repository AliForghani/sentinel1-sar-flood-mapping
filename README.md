# SAR Flood Detection Dashboard

> End-to-end Sentinel-1 SAR flood mapping — from scene discovery to georeferenced flood mask.

Built with Python and Streamlit. Search for radar scenes via the Alaska Satellite Facility API, register pre/post-flood images with SIFT+RANSAC, detect flooded pixels from backscatter change, and export analysis-ready GeoTIFFs.

![Python](https://img.shields.io/badge/Python-3.10+-blue) ![Streamlit](https://img.shields.io/badge/Streamlit-1.29+-red) ![Rasterio](https://img.shields.io/badge/Rasterio-1.3+-green) ![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## Table of Contents
- [Why SAR for Flood Detection](#why-sar-for-flood-detection)
- [SAR Background](#sar-background)
- [Setup](#setup)
- [App Workflow](#app-workflow)
- [Key Concepts](#key-concepts)
- [Output Files](#output-files)

---

## Why SAR for Flood Detection?

Floods happen during storms. Storms mean clouds. Clouds block optical satellites.

Sentinel-1 is a radar (SAR) satellite — it transmits microwave pulses, not light — so clouds are invisible to it. It sees the ground regardless of weather. This makes SAR the primary tool used in operational flood response systems worldwide.

**The physics is simple:**

| Surface | Radar behavior | Backscatter |
|---|---|---|
| Dry land | Bounces back toward satellite | High (bright) |
| Flood water | Flat surface reflects radar *away* | Low (dark) |

A before/after comparison reveals where backscatter dropped — that drop is the flood signal.

---

## SAR Background

> Reference: [How Sentinel-1 works](https://www.youtube.com/watch?v=GDOjlOGU4Co) — covers the 6-day revisit cycle and acquisition geometry.

Each Sentinel-1 image carries four key metadata fields used to ensure geometric consistency between acquisitions:

| Field | Description |
|---|---|
| **Relative orbit** | Think of these as named highways in the sky. Each orbit number corresponds to a fixed ground track. |
| **Orbit direction** | Ascending (south → north) or Descending (north → south). Affects the viewing angle significantly. |
| **Acquisition mode** | IW (Interferometric Wide Swath) is standard — 250 km swath, ~10 m resolution in GRD products. |
| **Polarization** | Orientation of the transmitted/received radar wave. VV and VH are the two channels in dual-pol products. |

### Matching Scenes for Change Detection

For before/after flood mapping, always match all four fields:

- ✅ Same relative orbit
- ✅ Same orbit direction
- ✅ Same acquisition mode
- ✅ Same polarization

This ensures any detected change is due to flooding, not a difference in how the satellite observed the scene. Because Sentinel-1's swath is ~250 km wide, a single location may appear in multiple orbits. For example:

```
New York City
├── Relative Orbit 42  (Ascending)
└── Relative Orbit 115 (Descending)
```

**Standard workflow:** find a good post-flood image first, note its four metadata fields, then search only for pre-flood images with the same fields. This app enforces that automatically by locking the relative orbit after post-flood scene selection.

This is why Sentinel-1 flood studies consistently report something like:
> *"All images were acquired in IW mode, descending pass, Relative Orbit 42, VV+VH polarization."*

---

## Setup

**Requirements:** Python 3.10+, conda recommended for rasterio on Windows.

```bash
# 1. Install rasterio via conda (pip often fails on Windows)
conda install -c conda-forge rasterio

# 2. Install remaining dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run app.py
```

**Data access:** Download Sentinel-1 GRD scenes from the [Alaska Satellite Facility](https://search.asf.alaska.edu). A free [NASA Earthdata account](https://urs.earthdata.nasa.gov) is required.

---

## App Workflow

| Step | What happens |
|---|---|
| **1 — Draw AOI** | Draw a rectangle on the interactive map. Save/load as JSON to reuse across sessions. Area is shown in km². |
| **2 — Filters** | Set flood date, acquisition mode (IW), orbit direction, and polarization. |
| **3 — Post-flood scene** | Search ASF within ±3 days of the flood date. Select a scene — its relative orbit locks automatically. |
| **4 — Pre-flood scene** | Search the same relative orbit over the preceding weeks. Select the best pre-flood acquisition. |
| **5 — Pair review** | Consistency check (orbit, direction, mode, polarization) + clickable download links for both scenes. |
| **6 — File paths** | Unzip the `.zip` → open `.SAFE/measurement/` → paste the full path to the `-vv-` tiff file. |
| **7 — Analysis** | Register images, detect flood pixels, display metrics and 4-panel visualization, export GeoTIFFs. |

---

## Key Concepts

### VV vs VH Polarization

This pipeline uses **VV only** (vertical–vertical).

- **VV** is sensitive to surface roughness and moisture — the primary flood signal. Water surface is flat → low VV backscatter.
- **VH** is sensitive to volume scattering from vegetation structure — less relevant for open-water detection.

### dB Conversion

Raw Sentinel-1 DN values span several orders of magnitude. Converting to decibels (dB) compresses this into a workable scale:

```
dB = 10 × log₁₀(backscatter)
```

Crucially, **differences in dB are multiplicative ratios** — a −3 dB drop means backscatter was halved, regardless of the absolute value. This makes the flood threshold physically meaningful across different scenes and locations.

### Speckle Noise

SAR images appear grainy due to coherent interference in the radar signal. This is not real texture — it is noise. A small uniform filter (kernel size 7) applied after dB conversion reduces speckle and lowers false-positive flood detections.

### Image Registration

Registration is a two-stage process in this pipeline.

**Stage 1 — Coarse alignment via GCPs**

Sentinel-1 GRD measurement TIFFs do not carry a standard affine geotransform. Instead, georeferencing is encoded as ~210 Ground Control Points (GCPs) scattered across the image — each GCP ties a pixel (row, col) to a geographic coordinate (lon, lat). `rasterio.vrt.WarpedVRT` reads these GCPs and fits a polynomial warp, resampling the image onto a regular EPSG:4326 grid. This gives both scenes a common coordinate frame and makes AOI clipping possible. For same-orbit pairs, this stage alone achieves sub-pixel alignment in most cases.

**Stage 2 — Fine registration via feature matching + homography**

Residual misalignment remains due to slight orbit deviations and terrain-induced distortions (SAR images are range-projected, not map-projected, so elevation causes pixel displacement). This is corrected with a classical CV pipeline:

1. **SIFT keypoints** — Scale-Invariant Feature Transform detects stable interest points at multiple scales. SAR images are first normalized to uint8 to meet SIFT's input requirement.
2. **Lowe's ratio test** — for each keypoint match, the distance to the best match must be < 0.75 × the distance to the second-best. This filters out ambiguous matches before RANSAC.
3. **BFMatcher (L2)** — brute-force nearest-neighbor matching in descriptor space. Used over FLANN here because the dataset is small enough that exact matching is fast.
4. **RANSAC homography** — `cv2.findHomography(..., cv2.RANSAC, 4.0)` fits a 3×3 projective transformation to the inlier matches. RANSAC iteratively samples 4-point minimal sets and keeps the model with the most inliers within a 4-pixel reprojection threshold. The result is robust to the ~30–50% outlier rate typical in SAR feature matching.
5. **`cv2.warpPerspective`** — applies the homography to the pre-flood image, warping it into the coordinate frame of the post-flood image.

**Why homography, not just translation?**

A pure translation assumes the two images are identical except for a shift. A homography is a full projective transform (8 degrees of freedom) that also handles rotation, scale, and perspective distortion — necessary because even same-orbit SAR acquisitions have small orbit deviations that introduce non-uniform spatial offsets across the image.

**Quality metrics**

| Metric | What it means | Expected value |
|---|---|---|
| RANSAC inliers | Matches consistent with the estimated homography | > 20 for reliable registration |
| Inlier ratio | Fraction of matches kept after RANSAC | > 0.3; low ratio suggests poor texture |
| Shift X / Y | Pixel translation component of the homography | Near 0 for same-orbit pairs |
| Rotation | Rotation angle of the homography | < 1° for same-orbit pairs; > 2° suggests a problem |

---

## Output Files

All outputs are georeferenced (EPSG:4326) and open directly in QGIS or ArcGIS.

| File | Type | Description |
|---|---|---|
| `pre_flood_vv_db.tif` | float32 | Pre-flood VV backscatter in dB |
| `pre_flood_registered_db.tif` | float32 | Pre-flood aligned to post-flood geometry |
| `post_flood_vv_db.tif` | float32 | Post-flood VV backscatter in dB |
| `flood_mask.tif` | uint8 | Binary flood mask (1 = flood, 0 = no flood) |
