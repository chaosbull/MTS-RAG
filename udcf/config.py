# Copyright 2026 chaosbull
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Author: chaosbull
# https://github.com/chaosbull

"""Project paths, process-variable maps, and model hyperparameters.

Plant Excel files keep their original Chinese column names. English aliases
are used in figures, tables, and reports.
"""

from __future__ import annotations

import os
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parent

RESULTS_DIR = PROJECT_ROOT / "results" / "erythromycin_validation"
FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR = RESULTS_DIR / "tables"
STEPS_DIR = RESULTS_DIR / "steps"
CACHE_DIR = PROJECT_ROOT / "cache"
RAMAN_DIR = PROJECT_ROOT / "raman_data"

# Plant batch folder. Override with UDCF_DATA_DIR; default is a local relative path.
# Raw industrial files are not shipped with this repository.
_default_data = PROJECT_ROOT / "data" / "erythromycin"
ERYTHROMYCIN_DATA_FOLDER = Path(os.environ.get("UDCF_DATA_DIR", _default_data))

VAR_TIME = "时间(h)"
VAR_BATCH = "文件"
VAR_TARGET = "化学效价"
VAR_CUM_TARGET = "化学效价_批次累计"

NET_CHANGE_VARS = frozenset({"化学效价", "无菌水", "液化糖", "豆油", "丙醇"})

POTENCY_MIN_VALID = -2000.0
POTENCY_MAX_VALID = 2500.0

TOP_N_DRIVERS = 8
MIN_SAMPLES_PER_HOUR = 40
TIME_WINDOW_HOURS = 4
ALARM_QUANTILE_LOW = 0.01
ALARM_QUANTILE_HIGH = 0.99
ALARM_BAND_WIDTH_FACTOR = 0.5
PERTURBATION_SIGMA = 1.0
MIN_INFLUENCE_ABS = 0.01
GRID_RES = 48
COEVOLUTION_GRID = 80
HOUR_STRIDE = 2

TIME_COL_START = 1
VARIABLE_COL_INDEX = 0
VISCOSITY_SPIKE_RATIO = 1.5

# Disjoint biological blocks. Patterns match cleaned plant column names.
PROCESS_BLOCKS = {
    "oxygenation": {
        "label": "Oxygenation",
        "patterns": ["空气流量", "搅拌转速", "溶解氧", "电机电流", "罐压"],
    },
    "carbon_feed": {
        "label": "Carbon Feed",
        "patterns": ["液化糖", "丙醇", "豆油", "总糖", "还原糖", "无菌水"],
    },
    "nitrogen": {
        "label": "Nitrogen",
        "patterns": ["氨基氮"],
    },
    "rheology": {
        "label": "Rheology / Biomass",
        "patterns": ["粘度", "菌浓", "滤速"],
    },
    "environment": {
        "label": "Environment",
        "patterns": ["罐温", "pH离线", "pH在线"],
    },
}

RAMAN_SOFT_TARGETS = {
    "总糖": "Total Sugar",
    "还原糖": "Reducing Sugar",
    "氨基氮": "Amino Nitrogen",
    "化学效价": "Potency (d/h)",
}

VAR_ENGLISH = {
    "罐温": "Tank Temperature",
    "罐压": "Tank Pressure",
    "重量": "Weight",
    "空气流量": "Air Flow Rate",
    "搅拌转速": "Agitator Speed",
    "电机电流": "Motor Current",
    "溶解氧": "Dissolved Oxygen",
    "无菌水": "Sterile Water (d/h)",
    "液化糖": "Liquefied Sugar (d/h)",
    "豆油": "Soybean Oil (d/h)",
    "丙醇": "Propanol (d/h)",
    "pH在线值": "pH Online",
    "pH离线值": "pH Offline",
    "菌浓": "Biomass Concentration",
    "粘度": "Viscosity",
    "滤速": "Filtration Rate",
    "菌丝阶段": "Mycelium Stage",
    "无菌情况": "Sterility Status",
    "还原糖": "Reducing Sugar",
    "总糖": "Total Sugar",
    "氨基氮": "Amino Nitrogen",
    "化学效价": "Potency (d/h)",
    "时间(h)": "Time (h)",
    "化学效价_批次累计": "Cumulative Potency",
}

VAR_UNITS = {
    "Tank Temperature": "C",
    "Tank Pressure": "MPa",
    "Air Flow Rate": "m3/h",
    "Agitator Speed": "rpm",
    "Dissolved Oxygen": "%",
    "Viscosity": "s",
    "Potency (d/h)": "u/ml/h",
    "Cumulative Potency": "u/ml",
    "Propanol (d/h)": "L/h",
    "Soybean Oil (d/h)": "L/h",
    "Liquefied Sugar (d/h)": "L/h",
    "Total Sugar": "g/L",
    "Amino Nitrogen": "g/L",
}

COLORS = {
    "primary": "#1B4F72",
    "secondary": "#D35400",
    "accent": "#1E8449",
    "alarm_high": "#C0392B",
    "alarm_low": "#2471A3",
    "neutral": "#7F8C8D",
    "band": "#5DADE2",
    "band_fill_alpha": 0.18,
    "bg": "#FBFCFD",
    "grid": "#D5D8DC",
    "advantage": "#0E6655",
    "baseline": "#922B21",
}

CMAP_DIVERGING = ["#2166AC", "#F7F7F7", "#B2182B"]
CMAP_SEQUENTIAL = ["#FFFFCC", "#FD8D3C", "#BD0026"]
HEATMAP_STD_CLIP = 1.0
HEATMAP_ROBUST_PCT = 95.0

BLOCK_RIDGE_ALPHA = 5.0
META_RIDGE_ALPHA = 1.0
LOCAL_RIDGE_ALPHA = 2.0
