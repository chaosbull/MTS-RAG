# Hierarchical Time-Varying Statistical Control Framework

Spearman screening, disjoint block Ridge, meta fusion, time-local influence, full-horizon propagation, and dynamic safety bands for industrial fermentation.

**Author:** [chaosbull](https://github.com/chaosbull)  
**License:** [Apache-2.0](LICENSE)

---

## Requirements

- Python 3.10+
- Plant batch Excel files (not included). The published run used industrial erythromycin fermentation data from a production plant (**2000+ batches**). Each file is a wide sheet: row 0 is time (h), column 0 is the variable name, and the body holds measurements.

```bash
python -m pip install -r requirements.txt
```

---

## How to run

1. Put batch Excel files in `data/erythromycin/`, **or** set `UDCF_DATA_DIR` to that folder.
2. Optional: put Raman CSV/XLSX/NPY files in `raman_data/` (see `raman_data/README.md`).
3. From the repository root:

```bash
python run_erythromycin.py
```

Windows example:

```bash
set UDCF_DATA_DIR=path\to\plant_excel_folder
python run_erythromycin.py
```

Linux / macOS:

```bash
export UDCF_DATA_DIR=/path/to/plant_excel_folder
python run_erythromycin.py
```

Flags:

```bash
python run_erythromycin.py --force-reload   # rebuild the hourly cache from Excel
python run_erythromycin.py --max-hour 190   # crop fermentation time
```

Outputs go to `results/erythromycin_validation/` (`figures/`, `tables/`, `steps/`). An hourly cache is written to `cache/` (gitignored).

---

## Layout

```
run_erythromycin.py       entry point
udcf/                     modeling package
raman_data/               optional spectra
results/                  last validation figures and tables
```

---

Copyright 2026 [chaosbull](https://github.com/chaosbull)
