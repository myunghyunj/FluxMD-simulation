# FluxMD Algorithms

## √v Cylindrical Sampling
Uniform sampling on a cylinder of radius `R` requires the cumulative
probability `P(r) = r^2 / R^2`. Inverting with a uniform variate `u`
yields `r = R * sqrt(u)`, ensuring even surface coverage.
*Groove bisector is fixed to 160°; see crystallographic distribution in Fig. 1 of the FluxMD pre-print.*

## Groove Geometry
B-form DNA major and minor groove bisectors are separated by roughly
160°. Minor groove vectors originate about 80° from the base-pair plane.

| Parameter | Value |
| --------- | ----- |
| Minor radius | 6 Å |
| Major radius | 11 Å |
| Asymmetry | 160° |
| GC fraction → Δradius | ±0.25 Å |

Sequence-dependent minor groove width is corrected by ±0.5 Å per unit
GC fraction over a five-base window.

### Groove angle extraction from builder markers
Groove angles are derived from the stored groove marker points of the DNA builder. The central base pair provides representative minor and major vectors whose azimuthal angles define `(θ_minor, θ_major)`.

### Vectorised √u sampling – memory notes & CuPy hand‑off
The sampler generates arrays of angles and radii in a single batch. Oversampling by a factor of two ensures enough accepted points after applying the groove bias. If `cupy` is available the arrays are allocated on the GPU; otherwise NumPy is used. The algorithm avoids large temporary buffers to remain memory efficient.
