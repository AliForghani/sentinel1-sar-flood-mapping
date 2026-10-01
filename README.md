# SAR Flood Detection Dashboard

> End-to-end Sentinel-1 SAR flood mapping — from scene discovery to georeferenced flood mask.

Built with Python and Streamlit. Search for radar scenes via the Alaska Satellite Facility API, align pre/post-flood images on a shared grid and fine-register them with SIFT+RANSAC (or phase correlation), detect flooded pixels from backscatter change, and export analysis-ready GeoTIFFs.

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

**Stage 1 — Georeferencing (coarse alignment) via GCPs**

Sentinel-1 GRD measurement TIFFs do not carry a standard affine geotransform. Instead, they include a sparse grid of ~210 Ground Control Points (GCPs) — roughly 21 across × 10 down the image. Each GCP ties one pixel (row, col) to a geographic coordinate (lon, lat); the positions of all other pixels are interpolated between them.

These GCPs are not surveyed ground points. ESA's processor computes them from the satellite's orbit and timing, projected onto the Earth ellipsoid with only a coarse terrain height. The same grid is also listed in the product's `annotation/*.xml` file (`<geolocationGrid>`).

`rasterio.vrt.WarpedVRT` fits a polynomial through each scene's GCPs and resamples it **directly onto one shared AOI grid**:

- **Extent** — the bounding box of the AOI.
- **Pixel size** — the post-flood scene's native resolution.
- **CRS** — lon/lat (EPSG:4326).

Because both scenes are warped onto the *same* grid, the two arrays have identical size, and pixel (i, j) has the same map coordinate (lon, lat) in both. Whether it also shows the same **ground feature** in both depends on how accurate each scene's GCPs are — that is what Stage 2 checks and corrects.

Nodata is handled explicitly: GRD border pixels (value 0) and pixels outside the AOI polygon become NaN, and are excluded from smoothing, registration, flood detection and the flood percentage.

Each scene is still placed on the map **independently** — using its own GCPs — so a feature (e.g. a road crossing) can land a pixel or so apart in the two images:

- **Small orbit and timing differences** between the two acquisitions, inherited through the GCPs.
- **Uncorrected terrain** — the GCPs use only a coarse terrain height, so elevated areas are slightly displaced. For same-orbit pairs the displacement is nearly identical in both images, so it largely cancels out in change detection.

**Stage 2 — Fine registration (image-to-image)**

Stage 2 compares the two images directly and corrects any small residual misalignment. For same-orbit pairs on a shared grid this is typically well under 1–2 pixels. Two methods are available in the app:

| Method | Model | Role |
|---|---|---|
| **SIFT + RANSAC homography** (default) | Projective transform (8 parameters) | Automatic feature-based matching — the classical computer-vision approach |
| **Phase correlation** | Shift only (2 parameters) | Simpler baseline — useful to compare against SIFT |

**In short:**

- **SIFT + RANSAC (feature-based)** — finds distinctive points in each image (road crossings, field corners) and gives each a "fingerprint". Fingerprints are matched between the images, ambiguous matches are dropped, and RANSAC fits a transform to the matches that agree with each other — ignoring the wrong ones. The pre-flood image is then warped with that transform.
- **Phase correlation (whole-image)** — instead of individual points, compares the two images in the frequency domain, where a shift in space shows up as a phase difference. The correlation peak gives the offset directly, with sub-pixel precision.
- **Trade-off** — SIFT is more flexible (it can also model rotation and scale) but can match speckle noise and needs textured areas. Phase correlation is simpler, faster and more robust to noise, but handles shifts only — which is usually all a same-orbit pair needs. When both report a similar shift, the result can be trusted.

**Why not just trust the GCPs?** They are computed from orbit and timing, not measured on the ground, so they are accurate to about a pixel. Change detection compares pixel by pixel, so even a one-pixel misalignment creates false "changes" along every edge — roads, field boundaries, riverbanks.

Both methods share the same safety rules: the estimated transform is applied **only if it passes sanity checks** (otherwise the pre-flood image is left as is and a warning explains why), and areas brought in from outside the image become NaN (nodata) rather than zero, so they cannot be mistaken for flood.

**Method A — SIFT + RANSAC homography (automatic feature matching)**

1. **Keypoint mask** — keypoints are only searched for well inside valid data (the valid area shrunk by 7 px). The edge between image and nodata is a strong artificial feature that would otherwise dominate the matching.
2. **SIFT keypoints** — Scale-Invariant Feature Transform detects distinctive points (corners, edges, bright structures) at multiple scales and describes each with a 128-number descriptor. Images are first stretched to 0–255 (2nd–98th percentile of valid pixels), since SIFT needs 8-bit input.
3. **BFMatcher (L2)** — brute-force nearest-neighbour matching of descriptors; for each pre-flood keypoint, the two closest post-flood keypoints are returned.
4. **Lowe's ratio test** — keep a match only if the best candidate is clearly better than the second-best (distance < 0.75 × second-best by default). This removes ambiguous matches.
5. **RANSAC homography** — `cv2.findHomography(..., cv2.RANSAC, threshold)` repeatedly fits a 3×3 transform to random sets of 4 matches and keeps the one that agrees with the most matches (inliers) within the pixel threshold (default 4 px). This makes the fit robust to wrong matches.
6. **Sanity checks** — the homography is applied only if it has enough inliers (default ≥ 20), a shift below the *max accepted shift* (default 3 px), and a rotation below 1°.
7. **`cv2.warpPerspective`** — resamples the pre-flood image with the accepted homography.

*Things to watch with SIFT on SAR:*

- **Speckle** — speckle is random between acquisitions, so some keypoints land on noise. The 7×7 smoothing and Lowe's ratio test reduce this; RANSAC rejects most of the rest.
- **Flat or water-dominated AOIs** — few distinctive structures means few reliable matches. Prefer AOIs that include roads, field edges, or built-up areas.
- **More flexibility than needed** — a homography can also model rotation, scale and perspective, which same-orbit pairs barely have. That extra freedom can fit noise; the rotation and shift checks guard against it.

**Method B — Phase correlation (shift only)**

1. **Phase correlation** — `skimage.registration.phase_cross_correlation` compares the two images in the frequency domain and finds the shift that best lines them up, with sub-pixel precision (1/10 pixel). It uses the whole image rather than individual keypoints, so it is robust to speckle and needs no tuning.
2. **Sanity check** — the shift is applied only if it is below the *max accepted shift* (default 3 px).
3. **Apply** — the pre-flood image is shifted with bilinear interpolation.

**Comparing the two** — on a good same-orbit pair, both methods should report a similar small shift (within a fraction of a pixel). If they disagree noticeably, SIFT has probably locked onto noise or changed areas — check its inlier count and rotation.

**What Stage 2 cannot fix**

- **Terrain displacement** — it varies locally with elevation, so no single global transform can correct it. If it matters (mountainous AOIs, different orbits), use proper terrain correction with a DEM (e.g. ESA SNAP or `pyroSAR`) instead of GCP warping.
- **Large changes** — flooding changes the image itself. Both methods rely on the unchanged parts of the AOI; if most of the AOI is flooded, estimates become less reliable (and are more likely to be rejected by the sanity checks).

**Quality metrics**

| Metric | Method | What it means | Expected value |
|---|---|---|---|
| Shift X / Y | Both | Estimated residual offset of pre-flood relative to post-flood | < 1–2 px for same-orbit pairs |
| Matches | SIFT | Keypoint matches that passed Lowe's ratio test | Tens to hundreds |
| RANSAC inliers | SIFT | Matches consistent with the fitted homography | ≥ 20 for reliable registration |
| Inlier ratio | SIFT | Fraction of matches kept by RANSAC | Higher is better; very low means mostly wrong matches |
| Rotation | SIFT | Rotation in the homography | ~0° for same-orbit pairs; > 1° is rejected |
| Correlation error | Phase | Phase-correlation error (0 = perfect match, 1 = no match) | Lower is better; flooding itself raises it |
| Applied | Both | Whether the transform passed the sanity checks | Yes for a normal same-orbit pair |

---

## Output Files

All outputs share the same AOI grid (EPSG:4326), are deflate-compressed, and open directly in QGIS or ArcGIS.

| File | Type | Description |
|---|---|---|
| `pre_flood_vv_db.tif` | float32 | Pre-flood VV backscatter in dB, on the shared grid (nodata = NaN) |
| `pre_flood_registered_db.tif` | float32 | Pre-flood after the Stage 2 shift (identical to the above if no shift was applied) |
| `post_flood_vv_db.tif` | float32 | Post-flood VV backscatter in dB (nodata = NaN) |
| `flood_mask.tif` | uint8 | Flood mask (1 = flood, 0 = no flood, 255 = nodata) |
