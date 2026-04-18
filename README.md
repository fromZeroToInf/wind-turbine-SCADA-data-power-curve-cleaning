# Wind-Turbine-SCADA-Data-Power-Curve-Cleaning (wtpcc)
`wtpcc` is a lightweight Python package for power-curve-based cleaning of wind turbine SCADA data.

The package filters SCADA observations by comparing measured power values against an interpolated reference power curve. In addition to point-wise filtering, it supports simple window-based removal of neighboring samples around out-of-band observations. It also provides file-based batch processing for CSV and Parquet files with optional parallel execution.

## Features

- power-curve-based filtering for SCADA data
- linear interpolation of a reference power curve via `numpy.interp`
- configurable power margin around the reference power curve
- minimum wind-speed threshold
- window-based filtering using convolution.
- batch processing of CSV and Parquet files
- optional parallel processing with automatic job estimation based on available RAM

## Installation
```bash
pip install wtpcc
```

# From Source
```bash
pip install .
or
pdm install
or 
pip install -e .
```

