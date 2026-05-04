# py-osrm-revival

[![Build Wheels](https://github.com/nick-fournier/py-osrm/actions/workflows/build_wheels.yml/badge.svg)](https://github.com/nick-fournier/py-osrm/actions/workflows/build_wheels.yml)
[![PyPI](https://img.shields.io/pypi/v/py-osrm-revival)](https://pypi.org/project/py-osrm-revival/)

**py-osrm-revival is an actively maintained continuation of [py-osrm](https://github.com/gis-ops/py-osrm), providing Python bindings to [osrm-backend](https://github.com/Project-OSRM/osrm-backend) using [nanobind](https://github.com/wjakob/nanobind).**

> This project was forked from [gis-ops/py-osrm](https://github.com/gis-ops/py-osrm) which is now archived and unmaintained. Development continues here.

This package binds to **OSRM v6.0.0** backend and includes preprocessing functionality.

---

## Supported Platforms
Platform | Arch
---|---
Linux | x86_64
MacOS | x86_64
Windows | x86_64
---

## Installation
py-osrm-revival is supported on **CPython 3.9+**.

**Install from PyPI:**
```bash
pip install py-osrm-revival
```

**Or install from GitHub Releases:**

Download the appropriate wheel for your platform from [Releases](https://github.com/nick-fournier/py-osrm/releases):
```bash
pip install https://github.com/nick-fournier/py-osrm/releases/download/v0.1.0/py_osrm_revival-0.1.0-cp39-abi3-linux_x86_64.whl
```

**Development install (requires compilation, ~5-10 min):**
```bash
pip install git+https://github.com/nick-fournier/py-osrm.git@revival
```

**From source:**
```bash
git clone https://github.com/nick-fournier/py-osrm.git
cd py-osrm
git checkout revival
pip install .
```

> **Note:** Development and source installations require a C++ compiler (GCC/Clang) and CMake.

## Quick Start

```python
import osrm

# Load preprocessed data
py_osrm = osrm.OSRM("path/to/data.osrm")

# Calculate route - pass coordinates directly!
result = py_osrm.Route([(7.41337, 43.72956), (7.41546, 43.73077)])
print(result["routes"][0]["distance"])  # Distance in meters
```

## Usage

### Routing Services

**Route** - Calculate routes between coordinates:
```python
result = py_osrm.Route(
    coordinates=[(7.41337, 43.72956), (7.41546, 43.73077)],
    steps=True,
    geometries="geojson",
    annotations=["speed", "distance"]
)
```

**Table** - Calculate distance/duration matrices:
```python
result = py_osrm.Table(
    coordinates=[(7.41337, 43.72956), (7.41546, 43.73077), (7.41862, 43.73216)],
    annotations=["distance", "duration"]
)
# Access via result["distances"] and result["durations"]
```

**Nearest** - Find nearest road segment:
```python
result = py_osrm.Nearest(
    coordinates=[(7.41337, 43.72956)],
    number=3
)
```

See the [documentation](https://nick-fournier.github.io/py-osrm/) for Trip, Match, and Tile services.

### Bulk Processing (Concurrent)

Process multiple routes in parallel using ThreadPoolExecutor for significant speedup:

```python
import polars as pl
import osrm

# Initialize OSRM instance
py_osrm = osrm.OSRM("path/to/data.osrm")

# Create DataFrame with origin-destination pairs
df = pl.DataFrame({
    "origin_lon": [7.41337, 7.41862, 7.42150],
    "origin_lat": [43.72956, 43.73216, 43.73400],
    "dest_lon": [7.41546, 7.42000, 7.42300],
    "dest_lat": [43.73077, 43.73300, 43.73500]
})

# Process all routes concurrently (releases GIL for true parallelism)
results = osrm.bulk_route(py_osrm, df, steps=True, geometries="geojson")

# Results DataFrame includes: distance, duration, geometry, success, error
print(results.select(["distance", "duration", "success"]))
```

**Installation with bulk processing support:**
```bash
pip install py-osrm-revival[bulk]           # Polars only
pip install py-osrm-revival[bulk-progress]  # Polars + tqdm progress bar
```

**Key features:**
- True parallel execution (GIL released in C++)
- Works with Polars DataFrames or dict-of-lists
- Per-row parameter customization
- Automatic error handling with `fail_fast` option
- Progress tracking with `show_progress=True`
- Typically 3-4x faster than sequential processing on 4+ cores

### Preprocessing

Before routing, you must preprocess OpenStreetMap data. Built-in profiles include `car`, `bicycle`, and `foot`:

**CH (Contraction Hierarchies) - Fastest queries:**
```python
import osrm

# Extract road network from OSM data
osrm.extract("data.osm.pbf", profile="car", output_path="data")

# Build routing graph (CH)
osrm.contract("data")

# Ready to use
py_osrm = osrm.OSRM("data.osrm")
```

**Using a custom profile:**
```python
osrm.extract("data.osm.pbf", profile="/path/to/custom.lua", output_path="data")
```

**CLI usage:**
```bash
python -m osrm extract data.osm.pbf --profile car
python -m osrm contract data
```

For **MLD (Multi-Level Dijkstra)** on larger datasets, see [preprocessing documentation](https://nick-fournier.github.io/py-osrm/).

## Advanced Usage

### Legacy Parameter Objects

<details>
<summary>Alternative usage with parameter objects (click to expand)</summary>

Parameter objects are still supported for backwards compatibility:

```python
# Create parameter object
route_params = osrm.RouteParameters(
    coordinates=[(7.41337, 43.72956), (7.41546, 43.73077)]
)

# Pass to Route method
result = py_osrm.Route(route_params)
```

This pattern is useful when constructing parameters programmatically or reusing them across multiple requests. However, the direct keyword argument pattern shown above is now recommended for most use cases.

</details>

---

## Documentation
[Documentation Page](https://nick-fournier.github.io/py-osrm/)

## Acknowledgments
This project is a continuation of [gis-ops/py-osrm](https://github.com/gis-ops/py-osrm). Thanks to the original authors for their foundational work.
