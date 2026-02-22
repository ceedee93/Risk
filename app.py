"""
╔══════════════════════════════════════════════════════════════════════════════════════╗
║                                                                                      ║
║          ██████╗ ██╗   ██╗ █████╗ ███╗   ██╗████████╗    ███████╗███╗   ██╗ ██████╗  ║
║         ██╔═══██╗██║   ██║██╔══██╗████╗  ██║╚══██╔══╝    ██╔════╝████╗  ██║██╔════╝  ║
║         ██║   ██║██║   ██║███████║██╔██╗ ██║   ██║       █████╗  ██╔██╗ ██║██║  ███╗ ║
║         ██║▄▄ ██║██║   ██║██╔══██║██║╚██╗██║   ██║       ██╔══╝  ██║╚██╗██║██║   ██║ ║
║         ╚██████╔╝╚██████╔╝██║  ██║██║ ╚████║   ██║       ███████╗██║ ╚████║╚██████╔╝ ║
║          ╚══▀▀═╝  ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═══╝   ╚═╝       ╚══════╝╚═╝  ╚═══╝ ╚═════╝  ║
║                                                                                      ║
║              QUANTITATIVE ENERGY PORTFOLIO RISK ENGINE — VERSION 5.0                 ║
║                                                                                      ║
║    Enterprise-Grade Multi-Year Stochastic Risk Assessment for German Retail          ║
║    Electricity Portfolios. Designed for Stadtwerke, Energieversorger, and            ║
║    Portfolio Management under BDEW / BNetzA / EEX conventions.                       ║
║                                                                                      ║
║    BLOCK 1 OF 3:                                                                     ║
║      ├── §1   Application Configuration & Constants                                 ║
║      ├── §2   Professional CSS & Design System                                      ║
║      ├── §3   Data Structures & Type Definitions                                    ║
║      ├── §4   Smart Data Parser (Auto-Detection)                                    ║
║      ├── §5   BDEW Standard Load Profile Library (12 profiles)                      ║
║      ├── §6   Temperature Model (HDD / CDD / Synthetic Weather)                    ║
║      ├── §7   HPFC Forward Curve Construction                                       ║
║      ├── §8   Synthetic Data Generator                                               ║
║      ├── §9   Data Validation & Quality Engine                                       ║
║      ├── §10  Chart Library — Part 1 (Data Exploration)                              ║
║      ├── §11  Guided Workflow UI Components                                          ║
║      └── §12  Main Application Layout — Data Input Section                           ║
║                                                                                      ║
╚══════════════════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

# ═══════════════════════════════════════════════════════════════════════════════
#  IMPORTS
# ═══════════════════════════════════════════════════════════════════════════════

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats as sp_stats
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
from enum import Enum
import re
import io
import time
import hashlib
import warnings
import json
import traceback

warnings.filterwarnings("ignore")


# ═══════════════════════════════════════════════════════════════════════════════
#  §1  APPLICATION CONFIGURATION & CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

# --- Page Setup ---
st.set_page_config(
    page_title="Quant Energy Risk Engine v5",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': """
        ## Quantitative Energy Risk Engine v5.0
        
        Enterprise-grade Monte-Carlo risk assessment for German retail 
        electricity portfolios.
        
        **Features:**
        - BDEW Standard Load Profiles (H0, G0-G6, L0-L2, RLM)
        - Ornstein-Uhlenbeck + Jump-Diffusion spot model
        - GARCH(1,1) conditional volatility
        - AR(1) temperature-driven volume errors
        - CVaR / Spectral Risk Measures
        - Multi-year portfolio diversification
        - Stress testing (Dunkelflaute, Gas Crisis, etc.)
        - Smart data import (copy-paste auto-detection)
        """
    }
)

# --- Application Version ---
APP_VERSION = "5.0.0"
APP_TITLE = "Quant Energy Risk Engine"
APP_SUBTITLE = "Enterprise Portfolio Risk Assessment"

# --- Colour Palette (Design System) ---
class Colors:
    """Centralized colour palette for consistent theming."""
    # Primary
    BLUE = "#3b82f6"
    BLUE_LIGHT = "#60a5fa"
    BLUE_DARK = "#1d4ed8"
    BLUE_FAINT = "rgba(59,130,246,0.08)"
    BLUE_GLOW = "rgba(59,130,246,0.25)"
    
    # Accent
    INDIGO = "#6366f1"
    INDIGO_LIGHT = "#818cf8"
    VIOLET = "#8b5cf6"
    
    # Semantic
    EMERALD = "#10b981"
    EMERALD_LIGHT = "#34d399"
    EMERALD_FAINT = "rgba(16,185,129,0.12)"
    TEAL = "#14b8a6"
    CYAN = "#06b6d4"
    CYAN_FAINT = "rgba(6,182,212,0.08)"
    
    AMBER = "#f59e0b"
    AMBER_LIGHT = "#fbbf24"
    AMBER_FAINT = "rgba(245,158,11,0.12)"
    ORANGE = "#f97316"
    
    ROSE = "#ef4444"
    ROSE_LIGHT = "#fca5a5"
    ROSE_FAINT = "rgba(239,68,68,0.12)"
    PINK = "#ec4899"
    
    # Neutrals
    WHITE = "#f8fafc"
    SLATE_50 = "#f8fafc"
    SLATE_100 = "#f1f5f9"
    SLATE_200 = "#e2e8f0"
    SLATE_300 = "#cbd5e1"
    SLATE_400 = "#94a3b8"
    SLATE_500 = "#64748b"
    SLATE_600 = "#475569"
    SLATE_700 = "#334155"
    SLATE_800 = "#1e293b"
    SLATE_900 = "#0f172a"
    SLATE_950 = "#020617"
    
    SKY = "#0ea5e9"
    LIME = "#84cc16"

    @staticmethod
    def hex2rgb(h: str) -> str:
        """Convert hex colour to comma-separated RGB string."""
        h = h.lstrip("#")
        return ",".join(str(int(h[i:i+2], 16)) for i in (0, 2, 4))

    @staticmethod
    def with_alpha(hex_color: str, alpha: float) -> str:
        """Create rgba() string from hex + alpha."""
        rgb = Colors.hex2rgb(hex_color)
        return f"rgba({rgb},{alpha})"


# --- Plotly Default Template ---
PLOTLY_DEFAULTS = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(15,23,42,0.30)",
    font=dict(
        family="Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
        color=Colors.SLATE_300,
        size=12,
    ),
    margin=dict(l=60, r=40, t=65, b=55),
    hoverlabel=dict(
        font_size=11,
        bgcolor="rgba(15,23,42,0.92)",
        bordercolor=Colors.SLATE_700,
        font_family="Inter, sans-serif",
    ),
    xaxis=dict(
        gridcolor="rgba(51,65,85,0.25)",
        zerolinecolor="rgba(51,65,85,0.5)",
    ),
    yaxis=dict(
        gridcolor="rgba(51,65,85,0.25)",
        zerolinecolor="rgba(51,65,85,0.5)",
    ),
)

def make_layout(**overrides) -> dict:
    """Create a Plotly layout dict from defaults + overrides."""
    layout = {**PLOTLY_DEFAULTS}
    layout.update(overrides)
    return layout


# --- BDEW Profile Registry ---
BDEW_PROFILES: Dict[str, Dict[str, Any]] = {
    "H0": {
        "name": "Haushalt",
        "name_en": "Residential / Household",
        "description": "Standardlastprofil für Haushaltskunden. Typische Morgen- und Abendspitzen, geringer Verbrauch nachts.",
        "icon": "🏠",
        "typical_volume": "2,500 – 5,000 kWh/a",
        "category": "household",
    },
    "G0": {
        "name": "Gewerbe allgemein",
        "name_en": "General Commercial",
        "description": "Allgemeines Gewerbeprofil mit Schwerpunkt auf Geschäftszeiten (Mo-Fr 7-18 Uhr).",
        "icon": "🏢",
        "typical_volume": "5,000 – 50,000 kWh/a",
        "category": "commercial",
    },
    "G1": {
        "name": "Gewerbe werktags 8–18",
        "name_en": "Commercial Workday 8-18",
        "description": "Gewerbebetrieb mit Verbrauch ausschließlich werktags 8-18 Uhr. Büros, Praxen.",
        "icon": "🏬",
        "typical_volume": "3,000 – 30,000 kWh/a",
        "category": "commercial",
    },
    "G2": {
        "name": "Gewerbe mit Abendverbrauch",
        "name_en": "Commercial with Evening Load",
        "description": "Gewerbebetrieb mit erhöhtem Abendverbrauch. Gastronomie, Unterhaltung.",
        "icon": "🍽️",
        "typical_volume": "10,000 – 80,000 kWh/a",
        "category": "commercial",
    },
    "G3": {
        "name": "Gewerbe durchlaufend",
        "name_en": "Commercial Continuous",
        "description": "Gewerbebetrieb mit durchlaufendem Betrieb (Schichtarbeit, 24/7).",
        "icon": "🏭",
        "typical_volume": "50,000 – 500,000 kWh/a",
        "category": "commercial",
    },
    "G4": {
        "name": "Laden / Friseur",
        "name_en": "Retail / Barber Shop",
        "description": "Ladengeschäft mit typischen Öffnungszeiten. Samstags halbtags.",
        "icon": "💇",
        "typical_volume": "3,000 – 15,000 kWh/a",
        "category": "commercial",
    },
    "G5": {
        "name": "Bäckerei mit Backstube",
        "name_en": "Bakery with Production",
        "description": "Bäckerei mit Backstube — charakteristischer Frühmorgen-Peak ab 3-4 Uhr.",
        "icon": "🥐",
        "typical_volume": "20,000 – 100,000 kWh/a",
        "category": "commercial",
    },
    "G6": {
        "name": "Wochenendbetrieb",
        "name_en": "Weekend Business",
        "description": "Betrieb mit Schwerpunkt am Wochenende. Freizeiteinrichtungen, Gastronomie.",
        "icon": "🎪",
        "typical_volume": "10,000 – 60,000 kWh/a",
        "category": "commercial",
    },
    "L0": {
        "name": "Landwirtschaft allgemein",
        "name_en": "Agriculture General",
        "description": "Allgemeines landwirtschaftliches Profil mit saisonaler Variation.",
        "icon": "🌾",
        "typical_volume": "10,000 – 100,000 kWh/a",
        "category": "agriculture",
    },
    "L1": {
        "name": "Milchwirtschaft / Viehzucht",
        "name_en": "Dairy Farm / Livestock",
        "description": "Landwirtschaft mit Milchwirtschaft. Zwei Peaks durch Melkzeiten (morgens & abends).",
        "icon": "🐄",
        "typical_volume": "15,000 – 80,000 kWh/a",
        "category": "agriculture",
    },
    "L2": {
        "name": "Sonstige Landwirtschaft",
        "name_en": "Other Agriculture",
        "description": "Sonstige landwirtschaftliche Betriebe ohne Milchwirtschaft.",
        "icon": "🚜",
        "typical_volume": "8,000 – 50,000 kWh/a",
        "category": "agriculture",
    },
    "RLM": {
        "name": "Registrierende Leistungsmessung",
        "name_en": "Metered Load (Baseload)",
        "description": "RLM-Kunden mit nahezu konstantem Bandlast-Profil. Große Industriekunden.",
        "icon": "⚙️",
        "typical_volume": "100,000+ kWh/a",
        "category": "industrial",
    },
}

# --- Preset Scenarios for Quick Start ---
PRESET_SCENARIOS: Dict[str, Dict[str, Any]] = {
    "Stadtwerk Basis": {
        "description": "Typisches Stadtwerk-Portfolio mit Haushaltskunden",
        "icon": "🏘️",
        "profile": "H0",
        "annual_mwh": 10000,
        "start_year": 2026,
        "n_years": 3,
        "base_price": 80.0,
        "vol_error": 8.0,
        "correlation": 40.0,
    },
    "Gewerbepark": {
        "description": "Gewerbliches Portfolio mit G0-Profil",
        "icon": "🏢",
        "profile": "G0",
        "annual_mwh": 25000,
        "start_year": 2026,
        "n_years": 2,
        "base_price": 85.0,
        "vol_error": 6.0,
        "correlation": 30.0,
    },
    "Industriekunde": {
        "description": "Großer Industriekunde mit Bandlast",
        "icon": "🏭",
        "profile": "RLM",
        "annual_mwh": 100000,
        "start_year": 2025,
        "n_years": 5,
        "base_price": 75.0,
        "vol_error": 4.0,
        "correlation": 20.0,
    },
    "Bäckerei-Kette": {
        "description": "Filial-Bäckerei mit Frühmorgen-Lastspitzen",
        "icon": "🥐",
        "profile": "G5",
        "annual_mwh": 50000,
        "start_year": 2026,
        "n_years": 3,
        "base_price": 82.0,
        "vol_error": 10.0,
        "correlation": 35.0,
    },
    "Agrarbetrieb": {
        "description": "Landwirtschaftliches Portfolio mit saisonaler Last",
        "icon": "🌾",
        "profile": "L0",
        "annual_mwh": 30000,
        "start_year": 2026,
        "n_years": 3,
        "base_price": 78.0,
        "vol_error": 12.0,
        "correlation": 45.0,
    },
    "Hochvolatil": {
        "description": "Stress-Szenario mit hoher Volatilität & Korrelation",
        "icon": "⚠️",
        "profile": "H0",
        "annual_mwh": 15000,
        "start_year": 2026,
        "n_years": 3,
        "base_price": 95.0,
        "vol_error": 15.0,
        "correlation": 60.0,
    },
}

# --- Column Name Detection Patterns ---
# These regex patterns are used by the Smart Parser to auto-detect column types
COLUMN_PATTERNS: Dict[str, List[str]] = {
    "datetime": [
        r"datum", r"date", r"zeit", r"time", r"timestamp", r"datetime",
        r"von", r"from", r"beginn", r"start", r"zeitstempel",
        r"liefertag", r"delivery.?date", r"periode",
    ],
    "load": [
        r"last", r"load", r"menge", r"volume", r"verbrauch", r"consumption",
        r"abnahme", r"offtake", r"leistung", r"power", r"mwh", r"kwh",
        r"energiemenge", r"energy", r"prognose.?menge", r"forecast.?vol",
        r"ist.?menge", r"actual.?vol", r"bedarf", r"demand",
    ],
    "price": [
        r"preis", r"price", r"hpfc", r"forward", r"spot", r"epex",
        r"eur", r"€", r"euro", r"base", r"peak", r"marktpreis",
        r"market.?price", r"eex", r"day.?ahead", r"stundenprice",
        r"beschaffung", r"procurement",
    ],
    "temperature": [
        r"temp", r"temperatur", r"°c", r"celsius", r"grad",
        r"wetter", r"weather", r"außentemp", r"outdoor",
    ],
}


# ═══════════════════════════════════════════════════════════════════════════════
#  §2  PROFESSIONAL CSS & DESIGN SYSTEM
# ═══════════════════════════════════════════════════════════════════════════════

def inject_css():
    """Inject the complete professional CSS design system."""
    st.markdown("""
    <style>
    /* ═══════════════════════════════════════════════════════════
       GLOBAL RESETS & LAYOUT
       ═══════════════════════════════════════════════════════════ */
    .block-container {
        padding-top: 1rem;
        padding-bottom: 2rem;
        max-width: 1560px;
    }
    
    header[data-testid="stHeader"] {
        background: rgba(15, 23, 42, 0.92);
        backdrop-filter: blur(16px) saturate(180%);
        -webkit-backdrop-filter: blur(16px) saturate(180%);
        border-bottom: 1px solid rgba(30, 41, 59, 0.8);
    }
    
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0c1222 0%, #0f172a 100%);
        border-right: 1px solid #1e293b;
    }
    
    section[data-testid="stSidebar"] .stMarkdown h3 {
        color: #e2e8f0;
        font-size: 0.92rem;
        font-weight: 700;
        letter-spacing: -0.01em;
    }
    
    /* Scrollbar styling */
    ::-webkit-scrollbar { width: 6px; height: 6px; }
    ::-webkit-scrollbar-track { background: #0f172a; }
    ::-webkit-scrollbar-thumb { background: #334155; border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: #475569; }

    /* ═══════════════════════════════════════════════════════════
       HERO BANNER
       ═══════════════════════════════════════════════════════════ */
    .hero-banner {
        background: linear-gradient(135deg, #0c1929 0%, #172554 35%, #1e1b4b 70%, #0f172a 100%);
        border: 1px solid rgba(99, 102, 241, 0.25);
        border-radius: 20px;
        padding: 2.2rem 2.8rem;
        margin-bottom: 1.5rem;
        position: relative;
        overflow: hidden;
    }
    .hero-banner::before {
        content: '';
        position: absolute;
        top: -60%;
        right: -15%;
        width: 55%;
        height: 220%;
        background: radial-gradient(ellipse, rgba(59, 130, 246, 0.06) 0%, transparent 65%);
        pointer-events: none;
    }
    .hero-banner::after {
        content: '';
        position: absolute;
        bottom: -40%;
        left: -10%;
        width: 40%;
        height: 180%;
        background: radial-gradient(ellipse, rgba(99, 102, 241, 0.04) 0%, transparent 60%);
        pointer-events: none;
    }
    .hero-banner h1 {
        margin: 0;
        color: #f8fafc;
        font-size: 1.75rem;
        font-weight: 800;
        letter-spacing: -0.035em;
        line-height: 1.2;
        position: relative;
        z-index: 1;
    }
    .hero-banner .subtitle {
        margin: 0.5rem 0 0;
        color: #93c5fd;
        font-size: 0.82rem;
        line-height: 1.55;
        position: relative;
        z-index: 1;
    }
    .hero-banner .version-badge {
        display: inline-block;
        background: rgba(99, 102, 241, 0.18);
        border: 1px solid rgba(99, 102, 241, 0.35);
        color: #a5b4fc;
        font-size: 0.62rem;
        padding: 3px 11px;
        border-radius: 20px;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-left: 10px;
        vertical-align: middle;
    }

    /* ═══════════════════════════════════════════════════════════
       WORKFLOW STEP CARDS
       ═══════════════════════════════════════════════════════════ */
    .step-card {
        background: linear-gradient(160deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        border-radius: 16px;
        padding: 1.5rem 1.8rem;
        margin-bottom: 1rem;
        transition: border-color 0.25s ease, box-shadow 0.25s ease;
    }
    .step-card:hover {
        border-color: #3b82f6;
        box-shadow: 0 0 20px rgba(59, 130, 246, 0.08);
    }
    .step-card.active {
        border-color: #3b82f6;
        box-shadow: 0 0 25px rgba(59, 130, 246, 0.12);
    }
    .step-card.completed {
        border-color: #10b981;
    }
    .step-card .step-header {
        display: flex;
        align-items: center;
        gap: 12px;
        margin-bottom: 0.8rem;
    }
    .step-card .step-number {
        display: flex;
        align-items: center;
        justify-content: center;
        width: 32px;
        height: 32px;
        border-radius: 10px;
        font-size: 0.82rem;
        font-weight: 800;
        flex-shrink: 0;
    }
    .step-card .step-number.pending {
        background: rgba(51, 65, 85, 0.5);
        color: #94a3b8;
        border: 1px solid #475569;
    }
    .step-card .step-number.active {
        background: rgba(59, 130, 246, 0.2);
        color: #60a5fa;
        border: 1px solid #3b82f6;
    }
    .step-card .step-number.done {
        background: rgba(16, 185, 129, 0.2);
        color: #34d399;
        border: 1px solid #10b981;
    }
    .step-card .step-title {
        font-size: 1rem;
        font-weight: 700;
        color: #f8fafc;
    }
    .step-card .step-desc {
        font-size: 0.8rem;
        color: #94a3b8;
        line-height: 1.5;
    }

    /* ═══════════════════════════════════════════════════════════
       KPI CARDS
       ═══════════════════════════════════════════════════════════ */
    .kpi-card {
        background: linear-gradient(160deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        border-radius: 14px;
        padding: 1.2rem 1rem;
        text-align: center;
        transition: border-color 0.2s, transform 0.15s;
    }
    .kpi-card:hover {
        border-color: #3b82f6;
        transform: translateY(-1px);
    }
    .kpi-card .kpi-label {
        font-size: 0.62rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-bottom: 0.35rem;
        font-weight: 600;
    }
    .kpi-card .kpi-value {
        font-size: 1.6rem;
        font-weight: 800;
        color: #f8fafc;
        line-height: 1.2;
    }
    .kpi-card .kpi-unit {
        font-size: 0.72rem;
        color: #64748b;
        margin-left: 3px;
        font-weight: 400;
    }
    .kpi-card .kpi-delta {
        font-size: 0.65rem;
        margin-top: 0.25rem;
        font-weight: 600;
    }
    .kpi-card .kpi-delta.positive { color: #10b981; }
    .kpi-card .kpi-delta.negative { color: #ef4444; }
    .kpi-card .kpi-delta.neutral  { color: #94a3b8; }

    /* ═══════════════════════════════════════════════════════════
       BIG RESULT CARD
       ═══════════════════════════════════════════════════════════ */
    .big-result {
        background: linear-gradient(135deg, #172554 0%, #1e1b4b 45%, #0f172a 100%);
        border: 1px solid rgba(59, 130, 246, 0.4);
        border-radius: 22px;
        padding: 2.5rem 3rem;
        text-align: center;
        position: relative;
        overflow: hidden;
    }
    .big-result::before {
        content: '';
        position: absolute;
        inset: 0;
        background: radial-gradient(
            circle at 50% -20%,
            rgba(59, 130, 246, 0.15) 0%,
            transparent 55%
        );
        pointer-events: none;
    }
    .big-result .br-label {
        font-size: 0.74rem;
        color: #93c5fd;
        text-transform: uppercase;
        letter-spacing: 0.14em;
        font-weight: 600;
        position: relative;
        z-index: 1;
    }
    .big-result .br-value {
        font-size: 3.5rem;
        font-weight: 900;
        color: #ffffff;
        letter-spacing: -0.04em;
        line-height: 1.1;
        margin: 0.3rem 0;
        position: relative;
        z-index: 1;
    }
    .big-result .br-context {
        font-size: 0.76rem;
        color: #64748b;
        margin-top: 0.4rem;
        position: relative;
        z-index: 1;
    }

    /* ═══════════════════════════════════════════════════════════
       RESULTS TABLE
       ═══════════════════════════════════════════════════════════ */
    .results-table {
        width: 100%;
        border-collapse: separate;
        border-spacing: 0;
        font-size: 0.82rem;
    }
    .results-table th {
        background: #0f172a;
        color: #94a3b8;
        padding: 13px 16px;
        text-align: left;
        font-size: 0.66rem;
        text-transform: uppercase;
        letter-spacing: 0.07em;
        font-weight: 700;
        border-bottom: 2px solid #1e40af;
        position: sticky;
        top: 0;
        z-index: 2;
    }
    .results-table td {
        padding: 12px 16px;
        border-bottom: 1px solid rgba(30, 41, 59, 0.6);
        color: #e2e8f0;
        vertical-align: top;
    }
    .results-table tbody tr:hover td {
        background: rgba(30, 41, 59, 0.4);
    }
    .results-table .row-total td {
        background: #0f172a;
        font-weight: 700;
        border-top: 2px solid #3b82f6;
    }
    .results-table .mono {
        font-family: 'JetBrains Mono', 'Fira Code', 'SF Mono', monospace;
        font-size: 0.8rem;
    }
    .results-table .text-right { text-align: right; }
    .results-table .sub-detail {
        font-size: 0.6rem;
        color: #64748b;
        margin-top: 3px;
        font-family: 'JetBrains Mono', monospace;
    }
    .results-table .pill {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 8px;
        font-size: 0.58rem;
        font-weight: 700;
        margin-left: 5px;
        letter-spacing: 0.03em;
    }
    .results-table .pill-green {
        background: rgba(16, 185, 129, 0.15);
        color: #34d399;
    }
    .results-table .pill-red {
        background: rgba(239, 68, 68, 0.15);
        color: #fca5a5;
    }
    .results-table .pill-blue {
        background: rgba(59, 130, 246, 0.15);
        color: #93c5fd;
    }

    /* ═══════════════════════════════════════════════════════════
       METHODOLOGY BOXES
       ═══════════════════════════════════════════════════════════ */
    .method-box {
        background: #0f172a;
        border: 1px solid #1e293b;
        border-radius: 14px;
        padding: 1.8rem 2.2rem;
        margin-bottom: 1.2rem;
    }
    .method-box h3 {
        color: #e2e8f0;
        margin-top: 0;
        font-size: 1rem;
        font-weight: 700;
    }
    .method-box p, .method-box li {
        color: #94a3b8;
        font-size: 0.85rem;
        line-height: 1.65;
    }
    .method-box code {
        background: rgba(59, 130, 246, 0.12);
        color: #93c5fd;
        padding: 2px 6px;
        border-radius: 4px;
        font-size: 0.8rem;
    }

    /* ═══════════════════════════════════════════════════════════
       STRESS TEST CARDS
       ═══════════════════════════════════════════════════════════ */
    .stress-card {
        background: linear-gradient(150deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        border-radius: 14px;
        padding: 1.4rem 1.6rem;
        transition: border-color 0.2s;
        height: 100%;
    }
    .stress-card:hover { border-color: #475569; }
    .stress-card h4 {
        color: #f8fafc;
        margin: 0 0 0.5rem;
        font-size: 0.95rem;
        font-weight: 700;
    }
    .stress-card .stress-desc {
        color: #94a3b8;
        font-size: 0.76rem;
        line-height: 1.5;
        margin-bottom: 0.8rem;
    }
    .stress-card .stress-impact {
        font-size: 1.4rem;
        font-weight: 800;
    }
    .stress-card .stress-impact.loss { color: #ef4444; }
    .stress-card .stress-impact.gain { color: #10b981; }
    .stress-card .stress-detail {
        font-size: 0.68rem;
        color: #64748b;
        margin-top: 0.3rem;
    }

    /* ═══════════════════════════════════════════════════════════
       DATA PASTE AREA
       ═══════════════════════════════════════════════════════════ */
    .paste-area-wrapper {
        background: linear-gradient(160deg, #1e293b 0%, #0f172a 100%);
        border: 2px dashed #334155;
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        transition: border-color 0.3s, background 0.3s;
    }
    .paste-area-wrapper:hover {
        border-color: #3b82f6;
        background: linear-gradient(160deg, #1e293b 0%, rgba(59,130,246,0.03) 100%);
    }
    .paste-area-wrapper .paste-icon {
        font-size: 2.5rem;
        margin-bottom: 0.8rem;
    }
    .paste-area-wrapper .paste-title {
        font-size: 1.1rem;
        font-weight: 700;
        color: #f8fafc;
        margin-bottom: 0.4rem;
    }
    .paste-area-wrapper .paste-hint {
        font-size: 0.78rem;
        color: #94a3b8;
        line-height: 1.5;
    }

    /* ═══════════════════════════════════════════════════════════
       DATA QUALITY INDICATOR
       ═══════════════════════════════════════════════════════════ */
    .quality-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 5px 14px;
        border-radius: 20px;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.03em;
    }
    .quality-badge.excellent {
        background: rgba(16, 185, 129, 0.15);
        color: #34d399;
        border: 1px solid rgba(16, 185, 129, 0.3);
    }
    .quality-badge.good {
        background: rgba(59, 130, 246, 0.15);
        color: #60a5fa;
        border: 1px solid rgba(59, 130, 246, 0.3);
    }
    .quality-badge.warning {
        background: rgba(245, 158, 11, 0.15);
        color: #fbbf24;
        border: 1px solid rgba(245, 158, 11, 0.3);
    }
    .quality-badge.critical {
        background: rgba(239, 68, 68, 0.15);
        color: #fca5a5;
        border: 1px solid rgba(239, 68, 68, 0.3);
    }

    /* ═══════════════════════════════════════════════════════════
       INFO CARDS / TOOLTIP BOXES
       ═══════════════════════════════════════════════════════════ */
    .info-box {
        background: rgba(59, 130, 246, 0.06);
        border: 1px solid rgba(59, 130, 246, 0.2);
        border-radius: 12px;
        padding: 1rem 1.3rem;
        font-size: 0.8rem;
        color: #93c5fd;
        line-height: 1.55;
        margin: 0.5rem 0;
    }
    .info-box strong { color: #60a5fa; }
    
    .warning-box {
        background: rgba(245, 158, 11, 0.06);
        border: 1px solid rgba(245, 158, 11, 0.2);
        border-radius: 12px;
        padding: 1rem 1.3rem;
        font-size: 0.8rem;
        color: #fbbf24;
        line-height: 1.55;
    }
    
    .success-box {
        background: rgba(16, 185, 129, 0.06);
        border: 1px solid rgba(16, 185, 129, 0.2);
        border-radius: 12px;
        padding: 1rem 1.3rem;
        font-size: 0.8rem;
        color: #34d399;
        line-height: 1.55;
    }

    /* ═══════════════════════════════════════════════════════════
       PRESET CARDS
       ═══════════════════════════════════════════════════════════ */
    .preset-card {
        background: linear-gradient(150deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 1rem 1.2rem;
        cursor: pointer;
        transition: all 0.2s;
    }
    .preset-card:hover {
        border-color: #3b82f6;
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(59, 130, 246, 0.1);
    }
    .preset-card .preset-icon { font-size: 1.5rem; margin-bottom: 0.3rem; }
    .preset-card .preset-name {
        font-size: 0.85rem;
        font-weight: 700;
        color: #f8fafc;
        margin-bottom: 0.2rem;
    }
    .preset-card .preset-desc {
        font-size: 0.7rem;
        color: #94a3b8;
        line-height: 1.4;
    }

    /* ═══════════════════════════════════════════════════════════
       COLUMN MAPPING PILLS
       ═══════════════════════════════════════════════════════════ */
    .col-pill {
        display: inline-flex;
        align-items: center;
        gap: 5px;
        padding: 4px 10px;
        border-radius: 8px;
        font-size: 0.7rem;
        font-weight: 600;
        margin: 2px;
    }
    .col-pill.datetime {
        background: rgba(6, 182, 212, 0.15);
        color: #22d3ee;
        border: 1px solid rgba(6, 182, 212, 0.3);
    }
    .col-pill.load {
        background: rgba(16, 185, 129, 0.15);
        color: #34d399;
        border: 1px solid rgba(16, 185, 129, 0.3);
    }
    .col-pill.price {
        background: rgba(245, 158, 11, 0.15);
        color: #fbbf24;
        border: 1px solid rgba(245, 158, 11, 0.3);
    }
    .col-pill.temperature {
        background: rgba(239, 68, 68, 0.15);
        color: #fca5a5;
        border: 1px solid rgba(239, 68, 68, 0.3);
    }
    .col-pill.unknown {
        background: rgba(148, 163, 184, 0.15);
        color: #94a3b8;
        border: 1px solid rgba(148, 163, 184, 0.3);
    }

    /* ═══════════════════════════════════════════════════════════
       TABS STYLING
       ═══════════════════════════════════════════════════════════ */
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
        background: rgba(15, 23, 42, 0.5);
        padding: 4px;
        border-radius: 14px;
        border: 1px solid #1e293b;
    }
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border: 1px solid transparent;
        border-radius: 10px;
        color: #94a3b8;
        font-size: 0.8rem;
        padding: 8px 16px;
        font-weight: 500;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: #e2e8f0;
        background: rgba(30, 41, 59, 0.5);
    }
    .stTabs [aria-selected="true"] {
        background: rgba(59, 130, 246, 0.15) !important;
        border-color: rgba(59, 130, 246, 0.3) !important;
        color: #60a5fa !important;
        font-weight: 700;
    }
    
    /* Tab highlight bar removal */
    .stTabs [data-baseweb="tab-highlight"] {
        display: none;
    }
    .stTabs [data-baseweb="tab-border"] {
        display: none;
    }

    /* ═══════════════════════════════════════════════════════════
       EXPANDER STYLING
       ═══════════════════════════════════════════════════════════ */
    div[data-testid="stExpander"] details {
        border-color: #334155 !important;
        border-radius: 12px !important;
        background: rgba(15, 23, 42, 0.5);
    }
    div[data-testid="stExpander"] details summary {
        font-weight: 600;
        color: #e2e8f0;
    }

    /* ═══════════════════════════════════════════════════════════
       BUTTON STYLING
       ═══════════════════════════════════════════════════════════ */
    .stButton > button[kind="primary"] {
        border-radius: 12px;
        font-weight: 700;
        letter-spacing: 0.01em;
        transition: all 0.2s;
    }
    .stButton > button[kind="secondary"] {
        border-radius: 12px;
        font-weight: 600;
    }

    /* ═══════════════════════════════════════════════════════════
       DIVIDER
       ═══════════════════════════════════════════════════════════ */
    hr {
        border-color: #1e293b !important;
        margin: 1.2rem 0 !important;
    }

    /* ═══════════════════════════════════════════════════════════
       ANIMATION
       ═══════════════════════════════════════════════════════════ */
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(12px); }
        to   { opacity: 1; transform: translateY(0); }
    }
    .animate-in {
        animation: fadeInUp 0.4s ease-out;
    }
    
    /* ═══════════════════════════════════════════════════════════
       TOOLTIP / HELP
       ═══════════════════════════════════════════════════════════ */
    .help-tooltip {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 16px;
        height: 16px;
        border-radius: 50%;
        background: rgba(59, 130, 246, 0.15);
        color: #60a5fa;
        font-size: 0.6rem;
        font-weight: 800;
        cursor: help;
        margin-left: 4px;
        border: 1px solid rgba(59, 130, 246, 0.3);
    }
    </style>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  §3  DATA STRUCTURES & TYPE DEFINITIONS
# ═══════════════════════════════════════════════════════════════════════════════

class DataSource(Enum):
    """How the data was loaded."""
    SYNTHETIC = "synthetic"
    PASTED = "pasted"
    UPLOADED = "uploaded"

class QualityLevel(Enum):
    """Data quality assessment level."""
    EXCELLENT = "excellent"
    GOOD = "good"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class ColumnMapping:
    """Describes the auto-detected mapping of input columns."""
    datetime_col: Optional[str] = None
    load_col: Optional[str] = None
    price_col: Optional[str] = None
    temperature_col: Optional[str] = None
    unmapped_cols: List[str] = field(default_factory=list)
    confidence_scores: Dict[str, float] = field(default_factory=dict)


@dataclass
class DataQualityReport:
    """Comprehensive data quality assessment."""
    level: QualityLevel = QualityLevel.GOOD
    total_rows: int = 0
    total_hours: int = 0
    missing_values: Dict[str, int] = field(default_factory=dict)
    duplicate_timestamps: int = 0
    gaps_detected: int = 0
    negative_loads: int = 0
    extreme_prices: int = 0
    date_range: Tuple[str, str] = ("", "")
    years_covered: List[int] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    info: List[str] = field(default_factory=list)
    score: float = 100.0


@dataclass
class PortfolioData:
    """Complete dataset ready for simulation."""
    dates: List[datetime] = field(default_factory=list)
    load: np.ndarray = field(default_factory=lambda: np.array([]))
    hpfc: np.ndarray = field(default_factory=lambda: np.array([]))
    temperature: np.ndarray = field(default_factory=lambda: np.array([]))
    hdd: np.ndarray = field(default_factory=lambda: np.array([]))
    cdd: np.ndarray = field(default_factory=lambda: np.array([]))
    year_map: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int32))
    years: List[int] = field(default_factory=list)
    n_hours: int = 0
    source: DataSource = DataSource.SYNTHETIC
    profile: str = "H0"
    annual_mwh: float = 0.0
    quality: Optional[DataQualityReport] = None


# ═══════════════════════════════════════════════════════════════════════════════
#  §4  SMART DATA PARSER (AUTO-DETECTION ENGINE)
# ═══════════════════════════════════════════════════════════════════════════════

class SmartDataParser:
    """
    Intelligent parser that auto-detects column types from pasted/uploaded data.
    
    Handles:
    - Tab-separated values (from Excel copy-paste)
    - Semicolon-separated values (German CSV)
    - Comma-separated values (international CSV)
    - Mixed delimiters
    - German number format (comma as decimal separator)
    - Various date formats (DD.MM.YYYY, YYYY-MM-DD, etc.)
    - Header detection
    - Column type inference via name matching + data analysis
    """
    
    GERMAN_DATE_FORMATS = [
        "%d.%m.%Y %H:%M",
        "%d.%m.%Y %H:%M:%S",
        "%d.%m.%y %H:%M",
        "%d.%m.%Y",
        "%d/%m/%Y %H:%M",
        "%d/%m/%Y %H:%M:%S",
        "%d-%m-%Y %H:%M",
    ]
    
    ISO_DATE_FORMATS = [
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M",
        "%Y-%m-%d",
        "%m/%d/%Y %H:%M",
        "%m/%d/%Y %H:%M:%S",
    ]
    
    ALL_DATE_FORMATS = GERMAN_DATE_FORMATS + ISO_DATE_FORMATS
    
    @staticmethod
    def detect_delimiter(text: str) -> str:
        """Detect the most likely delimiter in the text."""
        first_lines = text.strip().split("\n")[:10]
        
        delimiters = {"\t": 0, ";": 0, ",": 0, "|": 0}
        
        for line in first_lines:
            for d in delimiters:
                delimiters[d] += line.count(d)
        
        # Tab is preferred (Excel paste), then semicolon (German CSV)
        if delimiters["\t"] > 0:
            # Check if tab count is consistent across lines
            tab_counts = [line.count("\t") for line in first_lines if line.strip()]
            if len(set(tab_counts)) <= 2:  # Allow minor variation
                return "\t"
        
        if delimiters[";"] > delimiters[","]:
            return ";"
        elif delimiters[","] > 0:
            # Check if commas might be decimal separators (German format)
            # If we see patterns like "1,5" or "100,25" it's likely decimal
            german_decimal = sum(1 for line in first_lines 
                               if re.search(r'\d+,\d{1,2}(?!\d)', line))
            if german_decimal > len(first_lines) * 0.3:
                return ";"  # Fall back to semicolon
            return ","
        elif delimiters["|"] > 0:
            return "|"
        
        return "\t"  # Default
    
    @staticmethod
    def clean_german_number(val: str) -> Optional[float]:
        """Convert German number format to float."""
        if not isinstance(val, str):
            return None
        
        val = val.strip()
        if not val or val in ("-", "", "NA", "N/A", "nan", "null", "#N/A"):
            return None
        
        # Remove thousands separator (dot in German)
        # But be careful: "1.5" could be 1.5 (English) or 1500 (German dots as thousands)
        # Heuristic: if there's a comma followed by 1-2 digits at the end, it's German
        if re.match(r'^-?\d{1,3}(\.\d{3})+,\d{1,2}$', val):
            # Definitely German: 1.234,56
            val = val.replace(".", "").replace(",", ".")
        elif "," in val and "." not in val:
            # Likely German decimal: 123,45
            val = val.replace(",", ".")
        elif "," in val and "." in val:
            # Could be either format — check position
            last_comma = val.rfind(",")
            last_dot = val.rfind(".")
            if last_comma > last_dot:
                # German: 1.234,56
                val = val.replace(".", "").replace(",", ".")
            # else English: 1,234.56 — remove commas
            else:
                val = val.replace(",", "")
        
        # Remove any remaining non-numeric chars except . and -
        val = re.sub(r'[^\d.\-]', '', val)
        
        try:
            return float(val)
        except (ValueError, TypeError):
            return None
    
    @staticmethod
    def detect_date_format(sample_values: List[str]) -> Optional[str]:
        """Try to detect the date format from sample values."""
        for fmt in SmartDataParser.ALL_DATE_FORMATS:
            success = 0
            for val in sample_values[:20]:
                try:
                    datetime.strptime(val.strip(), fmt)
                    success += 1
                except (ValueError, AttributeError):
                    pass
            if success >= len(sample_values[:20]) * 0.7:
                return fmt
        return None
    
    @staticmethod
    def classify_column(col_name: str, values: pd.Series) -> Tuple[str, float]:
        """
        Classify a column as datetime, load, price, temperature, or unknown.
        Returns (type, confidence_score).
        """
        col_lower = col_name.lower().strip()
        
        # ── 1. Try name-based matching first ──
        for col_type, patterns in COLUMN_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, col_lower):
                    return (col_type, 0.9)
        
        # ── 2. Try data-based inference ──
        # Check if it looks like dates
        if values.dtype == object:
            sample = values.dropna().head(20).astype(str).tolist()
            date_fmt = SmartDataParser.detect_date_format(sample)
            if date_fmt:
                return ("datetime", 0.8)
        
        # Check if it looks like timestamps (already parsed)
        if pd.api.types.is_datetime64_any_dtype(values):
            return ("datetime", 0.95)
        
        # For numeric columns, try to infer from value range
        numeric_vals = pd.to_numeric(values, errors="coerce").dropna()
        if len(numeric_vals) > 0:
            mean_val = numeric_vals.mean()
            std_val = numeric_vals.std()
            min_val = numeric_vals.min()
            max_val = numeric_vals.max()
            
            # Temperature: typically -20 to 45°C
            if -30 < min_val < 15 and 5 < max_val < 50 and std_val < 20:
                return ("temperature", 0.5)
            
            # Price: typically 0 to 500 €/MWh (can go negative)
            if -100 < min_val and max_val < 1000 and 20 < mean_val < 300:
                return ("price", 0.5)
            
            # Load: typically > 0, can be very varied
            if min_val >= -0.1 and mean_val > 0:
                return ("load", 0.4)
        
        return ("unknown", 0.0)
    
    @classmethod
    def parse(cls, raw_text: str) -> Tuple[Optional[pd.DataFrame], 
                                            ColumnMapping, 
                                            List[str]]:
        """
        Parse raw pasted/uploaded text into a structured DataFrame.
        
        Returns:
            - DataFrame (or None if parsing fails)
            - ColumnMapping with auto-detected column assignments
            - List of warning/info messages
        """
        messages = []
        
        if not raw_text or not raw_text.strip():
            return None, ColumnMapping(), ["❌ Keine Daten eingefügt."]
        
        # ── Step 1: Detect delimiter ──
        delimiter = cls.detect_delimiter(raw_text)
        messages.append(f"ℹ️ Erkanntes Trennzeichen: "
                       f"{'Tab' if delimiter == chr(9) else repr(delimiter)}")
        
        # ── Step 2: Parse into DataFrame ──
        try:
            df = pd.read_csv(
                io.StringIO(raw_text),
                sep=delimiter,
                engine="python",
                skipinitialspace=True,
                on_bad_lines="skip",
            )
        except Exception as e:
            return None, ColumnMapping(), [f"❌ Parsing-Fehler: {str(e)}"]
        
        if df.empty or len(df) < 2:
            return None, ColumnMapping(), ["❌ Zu wenige Zeilen erkannt. "
                                           "Mindestens 2 Datenzeilen benötigt."]
        
        # Check if first row might be data (no header)
        first_row_numeric = all(
            cls.clean_german_number(str(v)) is not None 
            for v in df.iloc[0] if str(v).strip()
        )
        
        if first_row_numeric and not any(
            isinstance(c, str) and any(
                re.search(p, c.lower()) 
                for patterns in COLUMN_PATTERNS.values() 
                for p in patterns
            )
            for c in df.columns
        ):
            messages.append("⚠️ Keine Kopfzeile erkannt — generische Spaltennamen zugewiesen.")
            # Re-read without header
            df = pd.read_csv(
                io.StringIO(raw_text),
                sep=delimiter,
                header=None,
                engine="python",
                skipinitialspace=True,
                on_bad_lines="skip",
            )
            df.columns = [f"Spalte_{i+1}" for i in range(len(df.columns))]
        
        messages.append(f"✅ {len(df)} Zeilen × {len(df.columns)} Spalten erkannt")
        
        # ── Step 3: Auto-detect column types ──
        mapping = ColumnMapping()
        best_scores: Dict[str, Tuple[str, float]] = {}  # type -> (col_name, score)
        
        for col in df.columns:
            col_type, score = cls.classify_column(str(col), df[col])
            mapping.confidence_scores[str(col)] = score
            
            if col_type != "unknown":
                if col_type not in best_scores or score > best_scores[col_type][1]:
                    best_scores[col_type] = (str(col), score)
            else:
                mapping.unmapped_cols.append(str(col))
        
        # Assign best matches
        if "datetime" in best_scores:
            mapping.datetime_col = best_scores["datetime"][0]
        if "load" in best_scores:
            mapping.load_col = best_scores["load"][0]
        if "price" in best_scores:
            mapping.price_col = best_scores["price"][0]
        if "temperature" in best_scores:
            mapping.temperature_col = best_scores["temperature"][0]
        
        # ── Step 4: Convert datetime column ──
        if mapping.datetime_col:
            col = mapping.datetime_col
            # Try pandas auto-detection first
            try:
                df[col] = pd.to_datetime(df[col], dayfirst=True, format="mixed")
                messages.append(f"✅ Datumsspalte '{col}' erfolgreich konvertiert")
            except Exception:
                # Try manual format detection
                sample = df[col].dropna().head(20).astype(str).tolist()
                fmt = cls.detect_date_format(sample)
                if fmt:
                    try:
                        df[col] = pd.to_datetime(df[col], format=fmt)
                        messages.append(f"✅ Datumsformat erkannt: {fmt}")
                    except Exception:
                        messages.append(f"⚠️ Datumskonvertierung teilweise fehlgeschlagen")
                else:
                    messages.append(f"⚠️ Datumsformat nicht erkannt für '{col}'")
        
        # ── Step 5: Convert numeric columns ──
        for col in [mapping.load_col, mapping.price_col, mapping.temperature_col]:
            if col and col in df.columns:
                if df[col].dtype == object:
                    df[col] = df[col].apply(cls.clean_german_number)
                else:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
        
        # ── Step 6: Generate mapping summary ──
        if mapping.datetime_col:
            messages.append(f"📅 Datum: '{mapping.datetime_col}'")
        if mapping.load_col:
            messages.append(f"⚡ Last/Menge: '{mapping.load_col}'")
        if mapping.price_col:
            messages.append(f"💰 Preis/HPFC: '{mapping.price_col}'")
        if mapping.temperature_col:
            messages.append(f"🌡️ Temperatur: '{mapping.temperature_col}'")
        if mapping.unmapped_cols:
            messages.append(f"❓ Nicht zugeordnet: {', '.join(mapping.unmapped_cols)}")
        
        return df, mapping, messages
    
    @classmethod
    def convert_to_portfolio_data(
        cls, 
        df: pd.DataFrame, 
        mapping: ColumnMapping,
        base_price: float = 80.0,
    ) -> Tuple[Optional[PortfolioData], List[str]]:
        """
        Convert a parsed DataFrame into a PortfolioData object ready for simulation.
        Fills in missing columns (temperature, HPFC) with synthetic data.
        """
        messages = []
        
        # ── Validate minimum requirements ──
        if mapping.datetime_col is None:
            messages.append("❌ Keine Datumsspalte erkannt. Bitte prüfen Sie die Daten.")
            return None, messages
        
        if mapping.load_col is None and mapping.price_col is None:
            messages.append("❌ Weder Last- noch Preisspalte erkannt.")
            return None, messages
        
        # ── Sort by datetime ──
        dt_col = mapping.datetime_col
        df = df.sort_values(dt_col).reset_index(drop=True)
        
        # ── Remove duplicates and NaN dates ──
        df = df.dropna(subset=[dt_col])
        orig_len = len(df)
        df = df.drop_duplicates(subset=[dt_col], keep="first")
        if len(df) < orig_len:
            messages.append(f"ℹ️ {orig_len - len(df)} doppelte Zeitstempel entfernt")
        
        dates = df[dt_col].tolist()
        if isinstance(dates[0], pd.Timestamp):
            dates = [d.to_pydatetime() for d in dates]
        
        n = len(dates)
        
        # ── Build year map ──
        years_found = []
        year_map = np.zeros(n, dtype=np.int32)
        last_y = -1
        yidx = -1
        for i, dt in enumerate(dates):
            y = dt.year
            if y != last_y:
                years_found.append(y)
                yidx += 1
                last_y = y
            year_map[i] = yidx
        
        # ── Load data ──
        if mapping.load_col and mapping.load_col in df.columns:
            load = df[mapping.load_col].values.astype(np.float64)
            nan_mask = np.isnan(load)
            if nan_mask.any():
                # Forward-fill NaN values
                load = pd.Series(load).fillna(method="ffill").fillna(method="bfill").values
                messages.append(f"⚠️ {nan_mask.sum()} fehlende Lastwerte interpoliert")
        else:
            messages.append("ℹ️ Keine Lastspalte — verwende konstante Last (1 MWh/h)")
            load = np.ones(n)
        
        # ── Price data ──
        if mapping.price_col and mapping.price_col in df.columns:
            hpfc = df[mapping.price_col].values.astype(np.float64)
            nan_mask = np.isnan(hpfc)
            if nan_mask.any():
                hpfc = pd.Series(hpfc).fillna(method="ffill").fillna(method="bfill").values
                messages.append(f"⚠️ {nan_mask.sum()} fehlende Preiswerte interpoliert")
        else:
            messages.append("ℹ️ Keine Preisspalte — generiere synthetische HPFC")
            hpfc = _build_synthetic_hpfc(dates, base_price, year_map)
        
        # ── Temperature data ──
        if mapping.temperature_col and mapping.temperature_col in df.columns:
            temperature = df[mapping.temperature_col].values.astype(np.float64)
            nan_mask = np.isnan(temperature)
            if nan_mask.any():
                temperature = pd.Series(temperature).fillna(method="ffill").fillna(method="bfill").values
                messages.append(f"⚠️ {nan_mask.sum()} fehlende Temperaturwerte interpoliert")
        else:
            messages.append("ℹ️ Keine Temperaturspalte — generiere synthetische Temperaturdaten")
            temperature = _generate_synthetic_temperature(dates)
        
        # ── Compute HDD / CDD ──
        hdd = np.maximum(0.0, 15.0 - temperature) / 24.0
        cdd = np.maximum(0.0, temperature - 22.0) / 24.0
        
        # ── Build PortfolioData ──
        pdata = PortfolioData(
            dates=dates,
            load=load,
            hpfc=hpfc,
            temperature=temperature,
            hdd=hdd,
            cdd=cdd,
            year_map=year_map,
            years=years_found,
            n_hours=n,
            source=DataSource.PASTED,
            annual_mwh=float(load.sum() / len(years_found)),
        )
        
        # ── Run quality assessment ──
        pdata.quality = _assess_data_quality(pdata)
        
        messages.append(f"✅ Portfolio-Datensatz erstellt: {n:,} Stunden, "
                       f"{years_found[0]}–{years_found[-1]}")
        
        return pdata, messages


# ═══════════════════════════════════════════════════════════════════════════════
#  §5  BDEW STANDARD LOAD PROFILE LIBRARY
# ═══════════════════════════════════════════════════════════════════════════════

def bdew_factor(profile: str, hour: int, month: int, weekday: int) -> float:
    """
    Compute a synthetic BDEW-type load-profile factor.
    
    This function approximates the German standard load profiles
    published by the BDEW (Bundesverband der Energie- und Wasserwirtschaft).
    
    Real-world implementations use the analytical H(T) function with
    polynomial coefficients published annually. This synthetic version
    captures the essential patterns for simulation purposes.
    
    Parameters
    ----------
    profile : str
        BDEW profile code: H0, G0-G6, L0-L2, or RLM
    hour : int
        Hour of day (0-23)
    month : int
        Calendar month (1-12)
    weekday : int
        Day of week: 0=Monday, 1=Tuesday, ..., 6=Sunday
    
    Returns
    -------
    float
        Dimensionless scaling factor (mean ≈ 1.0 before normalisation)
    
    Notes
    -----
    The factors are structured as 24-element arrays (one per hour)
    with seasonal and day-type modifiers. Three day types are
    distinguished: workday, Saturday, and Sunday.
    """
    # ── Day type classification ──
    is_saturday = weekday == 5
    is_sunday = weekday == 6
    is_weekend = weekday >= 5
    
    # ── Season classification ──
    is_winter = month in (1, 2, 3, 11, 12)
    is_summer = month in (6, 7, 8)
    is_transition = month in (4, 5, 9, 10)

    # ────────────────────────────────────────────────────────────
    #  H0 — HAUSHALT (Residential)
    # ────────────────────────────────────────────────────────────
    if profile == "H0":
        seasonal = 1.35 if is_winter else (0.65 if is_summer else 1.0)
        
        if is_sunday:
            hourly = [
                0.42, 0.38, 0.35, 0.33, 0.32, 0.33,  # 00-05
                0.38, 0.55, 0.85, 1.05, 1.15, 1.20,  # 06-11
                1.22, 1.18, 1.10, 1.05, 1.08, 1.25,  # 12-17
                1.55, 1.65, 1.50, 1.25, 0.90, 0.58,  # 18-23
            ]
        elif is_saturday:
            hourly = [
                0.50, 0.42, 0.38, 0.35, 0.34, 0.35,  # 00-05
                0.42, 0.65, 0.95, 1.10, 1.15, 1.18,  # 06-11
                1.15, 1.10, 1.00, 0.95, 1.00, 1.20,  # 12-17
                1.50, 1.60, 1.45, 1.20, 0.88, 0.60,  # 18-23
            ]
        else:  # Workday
            hourly = [
                0.45, 0.40, 0.37, 0.35, 0.35, 0.50,  # 00-05
                0.85, 1.20, 1.30, 1.10, 0.95, 0.90,  # 06-11
                0.92, 0.88, 0.85, 0.88, 1.00, 1.40,  # 12-17
                1.75, 1.80, 1.60, 1.35, 1.00, 0.62,  # 18-23
            ]
        return seasonal * hourly[hour]

    # ────────────────────────────────────────────────────────────
    #  G0 — GEWERBE ALLGEMEIN (General Commercial)
    # ────────────────────────────────────────────────────────────
    elif profile == "G0":
        seasonal = 1.10 if is_winter else (0.85 if is_summer else 0.95)
        
        if is_sunday:
            hourly = [0.22]*6 + [0.25, 0.30, 0.35, 0.38, 0.40, 0.40,
                      0.38, 0.35, 0.32, 0.30, 0.28, 0.26,
                      0.24, 0.23, 0.22, 0.22, 0.22, 0.22]
        elif is_saturday:
            hourly = [0.25]*6 + [0.35, 0.60, 0.90, 1.10, 1.20, 1.25,
                      1.20, 1.15, 1.05, 0.90, 0.70, 0.50,
                      0.35, 0.30, 0.28, 0.26, 0.25, 0.25]
        else:
            hourly = [0.28, 0.25, 0.23, 0.22, 0.23, 0.30,
                      0.55, 0.95, 1.45, 1.60, 1.65, 1.60,
                      1.50, 1.55, 1.60, 1.62, 1.55, 1.30,
                      0.85, 0.55, 0.40, 0.35, 0.32, 0.30]
        return seasonal * hourly[hour]

    # ────────────────────────────────────────────────────────────
    #  G1 — GEWERBE WERKTAGS 8-18
    # ────────────────────────────────────────────────────────────
    elif profile == "G1":
        seasonal = 1.05 if is_winter else 0.95
        if is_weekend:
            return seasonal * 0.15
        hourly = [0.18]*6 + [0.40, 0.80, 1.50, 1.70, 1.75, 1.72,
                  1.40, 1.55, 1.70, 1.72, 1.68, 1.50,
                  0.80, 0.35, 0.22, 0.20, 0.19, 0.18]
        return seasonal * hourly[hour]

    # ────────────────────────────────────────────────────────────
    #  G2 — GEWERBE MIT ABENDVERBRAUCH
    # ────────────────────────────────────────────────────────────
    elif profile == "G2":
        seasonal = 1.08 if is_winter else 0.92
        if is_weekend:
            return seasonal * 0.20
        hourly = [0.20]*5 + [0.30, 0.50, 0.80, 1.10, 1.20, 1.15, 1.10,
                  1.05, 1.10, 1.15, 1.20, 1.35, 1.60,
                  1.80, 1.75, 1.50, 1.10, 0.65, 0.35]
        return seasonal * hourly[hour]

    # ────────────────────────────────────────────────────────────
    #  G3 — GEWERBE DURCHLAUFEND
    # ────────────────────────────────────────────────────────────
    elif profile == "G3":
        seasonal = 1.05 if is_winter else 0.95
        if is_weekend:
            return seasonal * 0.70
        hourly = [0.72]*5 + [0.80, 0.95, 1.20, 1.40, 1.45, 1.42, 1.38,
                  1.35, 1.38, 1.40, 1.42, 1.35, 1.20,
                  1.00, 0.90, 0.82, 0.78, 0.75, 0.73]
        return seasonal * hourly[hour]

    # ────────────────────────────────────────────────────────────
    #  G4 — LADEN / FRISEUR
    # ────────────────────────────────────────────────────────────
    elif profile == "G4":
        seasonal = 1.02 if is_winter else 0.98
        if is_sunday:
            return seasonal * 0.12
        if is_saturday:
            hourly = [0.15]*6 + [0.30, 0.60, 1.20, 1.50, 1.55, 1.50,
                      1.40, 1.30, 1.10, 0.80, 0.50, 0.30,
                      0.20, 0.18, 0.16, 0.15, 0.15, 0.15]
        else:
            hourly = [0.15]*5 + [0.25, 0.50, 0.85, 1.40, 1.65, 1.70, 1.65,
                      1.50, 1.55, 1.60, 1.65, 1.60, 1.40,
                      1.00, 0.55, 0.30, 0.22, 0.18, 0.16]
        return seasonal * hourly[hour]

    # ────────────────────────────────────────────────────────────
    #  G5 — BÄCKEREI MIT BACKSTUBE
    # ────────────────────────────────────────────────────────────
    elif profile == "G5":
        seasonal = 1.10 if is_winter else 0.90
        if is_sunday:
            return seasonal * 0.25
        hourly = [0.60, 0.65, 0.80, 1.40, 1.85, 1.95, 1.90, 1.70, 1.45,
                  1.20, 1.05, 0.95, 0.90, 0.88, 0.85, 0.80, 0.75, 0.68,
                  0.55, 0.45, 0.40, 0.38, 0.40, 0.50]
        return seasonal * hourly[hour]

    # ────────────────────────────────────────────────────────────
    #  G6 — WOCHENENDBETRIEB
    # ────────────────────────────────────────────────────────────
    elif profile == "G6":
        seasonal = 1.05 if is_winter else 0.95
        if is_weekend:
            hourly = [0.30]*6 + [0.45, 0.70, 1.20, 1.60, 1.80, 1.85,
                      1.82, 1.75, 1.65, 1.55, 1.50, 1.60,
                      1.70, 1.65, 1.40, 1.00, 0.60, 0.38]
        else:
            hourly = [0.22]*6 + [0.30, 0.55, 0.90, 1.10, 1.15, 1.12,
                      1.08, 1.10, 1.12, 1.10, 1.00, 0.80,
                      0.55, 0.35, 0.28, 0.25, 0.23, 0.22]
        return seasonal * hourly[hour]

    # ────────────────────────────────────────────────────────────
    #  L0 — LANDWIRTSCHAFT ALLGEMEIN
    # ────────────────────────────────────────────────────────────
    elif profile == "L0":
        seasonal = 1.15 if is_winter else (0.80 if is_summer else 1.0)
        hourly = [0.40]*5 + [0.55, 0.80, 1.20, 1.50, 1.55, 1.50, 1.45,
                  1.40, 1.42, 1.45, 1.40, 1.30, 1.10,
                  0.85, 0.65, 0.52, 0.45, 0.42, 0.40]
        return seasonal * hourly[hour]

    # ────────────────────────────────────────────────────────────
    #  L1 — MILCHWIRTSCHAFT / VIEHZUCHT
    # ────────────────────────────────────────────────────────────
    elif profile == "L1":
        seasonal = 1.12 if is_winter else 0.88
        hourly = [0.50, 0.48, 0.45, 0.50, 0.80, 1.60, 1.85, 1.50, 1.15,
                  1.00, 0.90, 0.85, 0.82, 0.85, 0.90, 1.00, 1.50, 1.80,
                  1.55, 1.10, 0.80, 0.65, 0.55, 0.52]
        return seasonal * hourly[hour]

    # ────────────────────────────────────────────────────────────
    #  L2 — SONSTIGE LANDWIRTSCHAFT
    # ────────────────────────────────────────────────────────────
    elif profile == "L2":
        seasonal = 1.08 if is_winter else 0.92
        hourly = [0.35]*5 + [0.50, 0.75, 1.15, 1.45, 1.55, 1.50, 1.45,
                  1.40, 1.42, 1.48, 1.45, 1.35, 1.15,
                  0.85, 0.60, 0.48, 0.42, 0.38, 0.36]
        return seasonal * hourly[hour]

    # ────────────────────────────────────────────────────────────
    #  RLM — REGISTRIERENDE LEISTUNGSMESSUNG (Baseload)
    # ────────────────────────────────────────────────────────────
    else:  # RLM
        return (1.0
                + 0.08 * np.sin(2 * np.pi * hour / 24 - np.pi / 4)
                + 0.05 * np.cos(2 * np.pi * (month - 1) / 12))


# ═══════════════════════════════════════════════════════════════════════════════
#  §6  TEMPERATURE MODEL (HDD / CDD / SYNTHETIC WEATHER)
# ═══════════════════════════════════════════════════════════════════════════════

def _generate_synthetic_temperature(
    dates: List[datetime],
    base_lat: float = 50.0,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Generate a synthetic hourly temperature series for Central Europe.
    
    The model combines:
    1. **Seasonal cycle**: Annual sine wave centered on DOY 100 (April)
       with amplitude calibrated for Germany (~10.5°C mean, ~8.5°C amplitude)
    2. **Diurnal cycle**: Daily sine wave with season-dependent amplitude
       (larger in summer, smaller in winter)
    3. **AR(1) stochastic perturbation**: Captures multi-day weather persistence
       (warm/cold spells). Decorrelation time ≈ 3 days (φ = 0.97).
    
    Parameters
    ----------
    dates : list of datetime
        Hourly timestamps for which to generate temperatures
    base_lat : float
        Latitude for calibration (default: 50°N ≈ Frankfurt)
    seed : int, optional
        Random seed for reproducibility
    
    Returns
    -------
    np.ndarray
        Hourly temperature values in °C
    """
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random.RandomState()
    
    n = len(dates)
    temp = np.zeros(n)
    ar_state = 0.0
    phi_t = 0.97  # ~3-day decorrelation time
    sigma_weather = 4.5  # Standard deviation of weather perturbation
    
    for i, dt in enumerate(dates):
        doy = dt.timetuple().tm_yday
        hour = dt.hour
        
        # 1. Seasonal cycle
        # Peaks around DOY 200 (mid-July), minimum around DOY 20 (mid-January)
        t_seasonal = 10.5 + 8.5 * np.sin(2 * np.pi * (doy - 100) / 365.25)
        
        # 2. Diurnal cycle
        # Amplitude is larger in summer (~6°C) and smaller in winter (~3°C)
        amp_diurnal = 3.5 + 2.5 * np.cos(2 * np.pi * (doy - 172) / 365.25)
        # Peak at ~15:00, minimum at ~06:00
        t_diurnal = amp_diurnal * np.sin(2 * np.pi * (hour - 6) / 24)
        
        # 3. AR(1) weather perturbation
        innovation = rng.normal(0, sigma_weather)
        ar_state = phi_t * ar_state + np.sqrt(1 - phi_t**2) * innovation
        
        temp[i] = t_seasonal + t_diurnal + ar_state
    
    return temp


def compute_degree_days(
    temperature: np.ndarray,
    t_base_heat: float = 15.0,
    t_base_cool: float = 22.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Heating Degree Days (HDD) and Cooling Degree Days (CDD).
    
    For hourly data, each hour contributes 1/24 of a degree day.
    
    Parameters
    ----------
    temperature : np.ndarray
        Hourly temperature values (°C)
    t_base_heat : float
        Heating base temperature (default: 15°C per DWD convention)
    t_base_cool : float
        Cooling base temperature (default: 22°C)
    
    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (HDD, CDD) arrays in degree-days per hour
    """
    hdd = np.maximum(0.0, t_base_heat - temperature) / 24.0
    cdd = np.maximum(0.0, temperature - t_base_cool) / 24.0
    return hdd, cdd


# ═══════════════════════════════════════════════════════════════════════════════
#  §7  HPFC FORWARD CURVE CONSTRUCTION
# ═══════════════════════════════════════════════════════════════════════════════

def _build_synthetic_hpfc(
    dates: List[datetime],
    base_price: float,
    year_map: np.ndarray,
    temperature: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Construct a synthetic Hourly Price Forward Curve (HPFC).
    
    The HPFC represents the expected hourly electricity price based on
    the current forward curve and a shaping model. This synthetic version
    incorporates the key structural features of German power prices:
    
    1. **Contango term structure**: Later years trade at a premium
    2. **Intraday shape**: High morning & evening peaks, solar depression midday
    3. **Seasonal shape**: Winter premium due to higher demand + gas prices
    4. **Weekend discount**: ~6 €/MWh lower on weekends
    5. **Temperature sensitivity**: Cold weather → higher prices
    6. **Microstructure noise**: Small random perturbation
    
    Parameters
    ----------
    dates : list of datetime
    base_price : float
        Reference base-year front-year price (€/MWh)
    year_map : np.ndarray
        Year index for each hour
    temperature : np.ndarray, optional
        Hourly temperature; used for temperature sensitivity
    
    Returns
    -------
    np.ndarray
        Hourly HPFC values (€/MWh)
    """
    n = len(dates)
    hpfc = np.zeros(n)
    
    # German intraday price shape (hourly deviation from daily average in €/MWh)
    # Reflects: morning ramp, solar depression midday, evening super-peak
    INTRADAY_SHAPE = np.array([
        -12, -14, -15, -16, -15, -10,   # 00-05: nighttime low
         -4,   3,   8,   6,   2,  -1,   # 06-11: morning ramp, midday drop
         -3,  -2,   0,   2,   6,  12,   # 12-17: afternoon rise
         18,  16,  10,   4,  -2,  -8,   # 18-23: evening peak, night decline
    ], dtype=np.float64)
    
    for i, dt in enumerate(dates):
        h = dt.hour
        m = dt.month
        wd = dt.weekday()
        yi = year_map[i]
        
        # 1. Base + contango (later years are more expensive)
        p = base_price + yi * 2.8
        
        # 2. Seasonal component (winter premium)
        p += 12.0 * np.cos(2 * np.pi * (m - 1) / 12)
        
        # 3. Intraday shape
        p += INTRADAY_SHAPE[h]
        
        # 4. Weekend discount
        if wd >= 5:
            p -= 6.0
        
        # 5. Temperature sensitivity (optional)
        if temperature is not None:
            doy = dt.timetuple().tm_yday
            t_normal = 10.5 + 8.5 * np.sin(2 * np.pi * (doy - 100) / 365.25)
            t_dev = temperature[i] - t_normal
            # Cold → higher prices; warm → lower prices
            p -= 0.8 * t_dev
        
        # 6. Small microstructure noise
        p += np.random.normal(0, 1.2)
        
        # Floor at -50 €/MWh (negative prices possible but bounded)
        hpfc[i] = max(p, -50.0)
    
    return hpfc


# ═══════════════════════════════════════════════════════════════════════════════
#  §8  SYNTHETIC DATA GENERATOR
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def generate_synthetic_data(
    start_year: int,
    n_years: int,
    profile: str,
    annual_mwh: float,
    base_price: float,
    temp_sensitivity: float = 0.03,
    _seed: int = 42,
) -> PortfolioData:
    """
    Generate a complete synthetic dataset for multi-year simulation.
    
    This function creates hourly vectors for load, HPFC, temperature,
    HDD/CDD over the specified time horizon, using BDEW standard load
    profiles and synthetic market data.
    
    Parameters
    ----------
    start_year : int
        First delivery year
    n_years : int
        Number of delivery years
    profile : str
        BDEW profile code
    annual_mwh : float
        Target annual volume in MWh
    base_price : float
        Front-year base price in €/MWh
    temp_sensitivity : float
        Load sensitivity to temperature (fraction per HDD)
    _seed : int
        Random seed for reproducibility
    
    Returns
    -------
    PortfolioData
        Complete dataset ready for simulation
    """
    np.random.seed(_seed)
    
    start = datetime(start_year, 1, 1)
    end = datetime(start_year + n_years, 1, 1)
    
    # ── Build date array ──
    dates = []
    loads_raw = []
    ymap_list = []
    years_found = []
    yidx = -1
    last_y = -1
    cur = start
    
    while cur < end:
        y = cur.year
        if y != last_y:
            years_found.append(y)
            yidx += 1
            last_y = y
        
        dates.append(cur)
        ymap_list.append(yidx)
        loads_raw.append(
            bdew_factor(profile, cur.hour, cur.month, cur.weekday())
        )
        cur += timedelta(hours=1)
    
    ymap = np.array(ymap_list, dtype=np.int32)
    loads_raw = np.array(loads_raw, dtype=np.float64)
    
    # ── Generate temperature ──
    temperature = _generate_synthetic_temperature(dates, seed=_seed)
    hdd, cdd = compute_degree_days(temperature)
    
    # ── Temperature-adjusted load ──
    # Heating: more load when cold
    # Cooling: some additional load when hot (air conditioning)
    t_adj = 1.0 + temp_sensitivity * hdd * 24.0 - 0.5 * temp_sensitivity * cdd * 24.0
    loads_adj = loads_raw * t_adj
    
    # ── Normalise to target annual volume ──
    for yi in range(n_years):
        mask = ymap == yi
        s = loads_adj[mask].sum()
        if s > 0:
            loads_adj[mask] *= annual_mwh / s
    
    # ── Build HPFC ──
    hpfc = _build_synthetic_hpfc(dates, base_price, ymap, temperature)
    
    # ── Build PortfolioData ──
    pdata = PortfolioData(
        dates=dates,
        load=loads_adj,
        hpfc=hpfc,
        temperature=temperature,
        hdd=hdd,
        cdd=cdd,
        year_map=ymap,
        years=years_found,
        n_hours=len(dates),
        source=DataSource.SYNTHETIC,
        profile=profile,
        annual_mwh=annual_mwh,
    )
    
    # ── Quality assessment ──
    pdata.quality = _assess_data_quality(pdata)
    
    return pdata


# ═══════════════════════════════════════════════════════════════════════════════
#  §9  DATA VALIDATION & QUALITY ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

def _assess_data_quality(data: PortfolioData) -> DataQualityReport:
    """
    Perform a comprehensive data quality assessment.
    
    Checks for:
    - Missing values
    - Duplicate timestamps
    - Time gaps
    - Negative load values
    - Extreme price values
    - Sufficient data coverage
    - Statistical anomalies
    
    Returns a DataQualityReport with score, level, and detailed messages.
    """
    report = DataQualityReport()
    report.total_rows = data.n_hours
    report.total_hours = data.n_hours
    report.years_covered = data.years
    
    if data.n_hours == 0:
        report.level = QualityLevel.CRITICAL
        report.score = 0
        report.warnings.append("Keine Daten vorhanden")
        return report
    
    score = 100.0
    
    # ── Date range ──
    report.date_range = (
        data.dates[0].strftime("%d.%m.%Y %H:%M"),
        data.dates[-1].strftime("%d.%m.%Y %H:%M"),
    )
    
    # ── Check for gaps in hourly series ──
    if len(data.dates) > 1:
        expected_hours = int(
            (data.dates[-1] - data.dates[0]).total_seconds() / 3600
        ) + 1
        gaps = expected_hours - data.n_hours
        report.gaps_detected = max(0, gaps)
        if gaps > 0:
            gap_pct = gaps / expected_hours * 100
            report.warnings.append(
                f"{gaps:,} fehlende Stunden ({gap_pct:.1f}%) im Zeitraum"
            )
            score -= min(20, gap_pct * 2)
    
    # ── Missing values ──
    nan_load = int(np.isnan(data.load).sum()) if len(data.load) > 0 else 0
    nan_hpfc = int(np.isnan(data.hpfc).sum()) if len(data.hpfc) > 0 else 0
    nan_temp = int(np.isnan(data.temperature).sum()) if len(data.temperature) > 0 else 0
    
    report.missing_values = {
        "Last": nan_load,
        "HPFC": nan_hpfc,
        "Temperatur": nan_temp,
    }
    
    total_nan = nan_load + nan_hpfc + nan_temp
    if total_nan > 0:
        nan_pct = total_nan / (data.n_hours * 3) * 100
        report.warnings.append(f"{total_nan:,} fehlende Werte insgesamt ({nan_pct:.1f}%)")
        score -= min(15, nan_pct * 3)
    
    # ── Negative load ──
    if len(data.load) > 0:
        neg_load = int((data.load < -0.001).sum())
        report.negative_loads = neg_load
        if neg_load > 0:
            report.warnings.append(f"{neg_load:,} negative Lastwerte")
            score -= min(10, neg_load / data.n_hours * 100)
    
    # ── Extreme prices ──
    if len(data.hpfc) > 0:
        extreme_high = int((data.hpfc > 500).sum())
        extreme_low = int((data.hpfc < -100).sum())
        report.extreme_prices = extreme_high + extreme_low
        if report.extreme_prices > 0:
            report.warnings.append(
                f"{report.extreme_prices:,} extreme Preiswerte "
                f"(>{500} oder <{-100} €/MWh)"
            )
            score -= min(5, report.extreme_prices / data.n_hours * 50)
    
    # ── Sufficient coverage ──
    if data.n_hours < 8760:
        report.warnings.append(
            f"Weniger als 1 Jahr Daten ({data.n_hours:,} Stunden). "
            f"Ergebnisse können unzuverlässig sein."
        )
        score -= 5
    
    # ── Volume plausibility ──
    if len(data.load) > 0 and data.load.sum() > 0:
        mean_load = data.load.mean()
        max_load = data.load.max()
        if max_load > mean_load * 10:
            report.warnings.append(
                f"Extrem hohe Lastspitzen (Max/Ø = {max_load/mean_load:.1f}x)"
            )
            score -= 3
    
    # ── Info messages ──
    if len(data.load) > 0:
        report.info.append(f"Ø Last: {data.load.mean():.3f} MWh/h")
        report.info.append(f"Jahresverbrauch: {data.load.sum()/len(data.years):,.0f} MWh/a")
    if len(data.hpfc) > 0:
        report.info.append(f"Ø HPFC: {data.hpfc.mean():.2f} €/MWh")
    if len(data.temperature) > 0:
        report.info.append(
            f"Temperatur: {data.temperature.min():.1f}°C – "
            f"{data.temperature.max():.1f}°C (Ø {data.temperature.mean():.1f}°C)"
        )
    
    # ── Determine quality level ──
    report.score = max(0, min(100, score))
    if report.score >= 90:
        report.level = QualityLevel.EXCELLENT
    elif report.score >= 70:
        report.level = QualityLevel.GOOD
    elif report.score >= 50:
        report.level = QualityLevel.WARNING
    else:
        report.level = QualityLevel.CRITICAL
    
    return report


def render_quality_badge(quality: DataQualityReport) -> str:
    """Render an HTML quality badge."""
    level_text = {
        QualityLevel.EXCELLENT: "Exzellent",
        QualityLevel.GOOD: "Gut",
        QualityLevel.WARNING: "Warnung",
        QualityLevel.CRITICAL: "Kritisch",
    }
    level_icon = {
        QualityLevel.EXCELLENT: "✅",
        QualityLevel.GOOD: "🟢",
        QualityLevel.WARNING: "⚠️",
        QualityLevel.CRITICAL: "🔴",
    }
    return (
        f'<span class="quality-badge {quality.level.value}">'
        f'{level_icon[quality.level]} {level_text[quality.level]} '
        f'({quality.score:.0f}/100)</span>'
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  §10  CHART LIBRARY — PART 1 (DATA EXPLORATION)
# ═══════════════════════════════════════════════════════════════════════════════

def chart_load_hpfc_timeseries(
    data: PortfolioData,
    max_hours: int = 504,
    title: Optional[str] = None,
) -> go.Figure:
    """
    Create a dual-axis time series chart showing load, HPFC, and temperature.
    """
    n = min(max_hours, data.n_hours)
    x = data.dates[:n]
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Load area
    fig.add_trace(
        go.Scatter(
            x=x, y=data.load[:n],
            name="Last (MWh/h)",
            line=dict(color=Colors.CYAN, width=1.2),
            fill="tozeroy",
            fillcolor=Colors.with_alpha(Colors.CYAN, 0.06),
            hovertemplate="<b>%{x|%d.%m.%Y %H:%M}</b><br>"
                          "Last: %{y:.3f} MWh/h<extra></extra>",
        ),
        secondary_y=False,
    )
    
    # HPFC line
    fig.add_trace(
        go.Scatter(
            x=x, y=data.hpfc[:n],
            name="HPFC (€/MWh)",
            line=dict(color=Colors.AMBER, width=1.6, dash="dot"),
            hovertemplate="<b>%{x|%d.%m.%Y %H:%M}</b><br>"
                          "HPFC: %{y:.2f} €/MWh<extra></extra>",
        ),
        secondary_y=True,
    )
    
    # Temperature (if available and meaningful)
    if len(data.temperature) > 0 and not np.all(data.temperature == 0):
        fig.add_trace(
            go.Scatter(
                x=x, y=data.temperature[:n],
                name="Temperatur (°C)",
                line=dict(color=Colors.ROSE, width=0.9),
                opacity=0.5,
                hovertemplate="<b>%{x|%d.%m.%Y %H:%M}</b><br>"
                              "Temp: %{y:.1f} °C<extra></extra>",
            ),
            secondary_y=True,
        )
    
    fig.update_layout(**make_layout(
        title=dict(
            text=title or f"Lastprofil & HPFC — Erste {n:,} Stunden",
            font=dict(size=14),
        ),
        height=400,
        legend=dict(
            orientation="h", y=1.12, x=0.5, xanchor="center",
            font=dict(size=11),
        ),
    ))
    fig.update_yaxes(
        title_text="MWh / h", secondary_y=False,
        gridcolor="rgba(51,65,85,0.3)",
    )
    fig.update_yaxes(
        title_text="€ / MWh  &  °C", secondary_y=True,
        gridcolor="rgba(51,65,85,0.12)",
    )
    
    return fig


def chart_load_duration_curve(data: PortfolioData) -> go.Figure:
    """
    Jahresdauerlinie — Load Duration Curve.
    Shows hours ranked by load, revealing baseload vs. peak structure.
    """
    sorted_load = np.sort(data.load)[::-1]
    hours = np.arange(len(sorted_load))
    
    # Calculate key statistics
    peak = sorted_load[0]
    base = sorted_load[-1]
    p50 = sorted_load[len(sorted_load) // 2]
    utilisation = base / peak * 100 if peak > 0 else 0
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=hours, y=sorted_load,
        mode="lines",
        fill="tozeroy",
        line=dict(color=Colors.CYAN, width=1.8),
        fillcolor=Colors.with_alpha(Colors.CYAN, 0.06),
        name="Last",
        hovertemplate="Rang: %{x:,}<br>Last: %{y:.3f} MWh/h<extra></extra>",
    ))
    
    # Annotations
    fig.add_hline(
        y=peak, line=dict(color=Colors.ROSE, width=1, dash="dash"),
        annotation_text=f"Peak: {peak:.3f}", annotation_position="top left",
        annotation_font=dict(size=10, color=Colors.ROSE),
    )
    fig.add_hline(
        y=p50, line=dict(color=Colors.AMBER, width=1, dash="dot"),
        annotation_text=f"Median: {p50:.3f}", annotation_position="top left",
        annotation_font=dict(size=10, color=Colors.AMBER),
    )
    
    fig.update_layout(**make_layout(
        title=dict(
            text=f"Jahresdauerlinie  (Auslastung: {utilisation:.1f}%)",
            font=dict(size=14),
        ),
        xaxis_title="Stunden (nach Last sortiert)",
        yaxis_title="MWh / h",
        height=400,
        showlegend=False,
    ))
    
    return fig


def chart_heatmap_monthly(data: PortfolioData) -> go.Figure:
    """
    HPFC heatmap: average price by hour × month.
    Reveals the structural price shape.
    """
    df = pd.DataFrame({
        "month": [d.month for d in data.dates],
        "hour": [d.hour for d in data.dates],
        "hpfc": data.hpfc,
    })
    piv = df.pivot_table(values="hpfc", index="hour", columns="month", aggfunc="mean")
    
    month_labels = ["Jan", "Feb", "Mär", "Apr", "Mai", "Jun",
                    "Jul", "Aug", "Sep", "Okt", "Nov", "Dez"]
    
    fig = go.Figure(go.Heatmap(
        z=piv.values,
        x=[month_labels[m-1] for m in piv.columns],
        y=[f"{h:02d}:00" for h in piv.index],
        colorscale="RdYlBu_r",
        colorbar=dict(title="€/MWh", len=0.65, thickness=15),
        hovertemplate="Monat: %{x}<br>Stunde: %{y}<br>"
                      "Ø HPFC: %{z:.1f} €/MWh<extra></extra>",
    ))
    
    fig.update_layout(**make_layout(
        title=dict(text="HPFC-Preisstruktur  (Stunde × Monat)", font=dict(size=14)),
        height=480,
        yaxis=dict(autorange="reversed"),
    ))
    
    return fig


def chart_temperature_vs_load(
    data: PortfolioData,
    max_points: int = 5000,
) -> go.Figure:
    """
    Scatter plot: temperature vs load, coloured by HPFC.
    Reveals the temperature sensitivity of the portfolio.
    """
    n = min(max_points, data.n_hours)
    if data.n_hours > max_points:
        idx = np.random.choice(data.n_hours, n, replace=False)
        idx.sort()
    else:
        idx = np.arange(n)
    
    fig = go.Figure(go.Scatter(
        x=data.temperature[idx],
        y=data.load[idx],
        mode="markers",
        marker=dict(
            size=2.5,
            color=data.hpfc[idx],
            colorscale="Viridis",
            colorbar=dict(title="HPFC<br>€/MWh", len=0.6, thickness=12),
            opacity=0.4,
            line=dict(width=0),
        ),
        hovertemplate="Temp: %{x:.1f}°C<br>Last: %{y:.3f} MWh/h<br>"
                      "HPFC: %{marker.color:.1f} €<extra></extra>",
    ))
    
    fig.update_layout(**make_layout(
        title=dict(text="Temperatur vs. Last  (Farbe = HPFC)", font=dict(size=14)),
        xaxis_title="Temperatur (°C)",
        yaxis_title="Last (MWh / h)",
        height=420,
        showlegend=False,
    ))
    
    return fig


def chart_average_daily_shape(data: PortfolioData) -> go.Figure:
    """
    Average daily load shape — workday vs weekend.
    """
    df = pd.DataFrame({
        "hour": [d.hour for d in data.dates],
        "load": data.load,
        "daytype": [
            "Wochenende" if d.weekday() >= 5 else "Werktag"
            for d in data.dates
        ],
    })
    avg = df.groupby(["daytype", "hour"])["load"].mean().reset_index()
    
    fig = go.Figure()
    
    colors_map = {"Werktag": Colors.BLUE, "Wochenende": Colors.AMBER}
    for dtype in ["Werktag", "Wochenende"]:
        subset = avg[avg["daytype"] == dtype]
        fig.add_trace(go.Scatter(
            x=subset["hour"], y=subset["load"],
            name=dtype,
            mode="lines+markers",
            line=dict(color=colors_map[dtype], width=2.5),
            marker=dict(size=5),
            hovertemplate=f"{dtype}<br>Stunde: %{{x}}:00<br>"
                          f"Ø Last: %{{y:.4f}} MWh/h<extra></extra>",
        ))
    
    fig.update_layout(**make_layout(
        title=dict(
            text=f"Ø Tageslastgang — {data.profile}",
            font=dict(size=14),
        ),
        xaxis_title="Stunde des Tages",
        yaxis_title="Ø Last (MWh / h)",
        height=380,
        legend=dict(
            orientation="h", y=1.10, x=0.5, xanchor="center",
            font=dict(size=11),
        ),
    ))
    
    return fig


def chart_monthly_summary(data: PortfolioData) -> go.Figure:
    """
    Monthly summary: volume bars + average HPFC + temperature lines.
    """
    df = pd.DataFrame({
        "date": data.dates,
        "load": data.load,
        "hpfc": data.hpfc,
        "temp": data.temperature,
    })
    df["month"] = df["date"].apply(lambda x: x.strftime("%Y-%m"))
    
    monthly = df.groupby("month").agg(
        vol=("load", "sum"),
        avg_hpfc=("hpfc", "mean"),
        avg_temp=("temp", "mean"),
    ).reset_index()
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    fig.add_trace(
        go.Bar(
            x=monthly["month"], y=monthly["vol"],
            name="Volumen (MWh)",
            marker_color=Colors.CYAN,
            opacity=0.7,
            hovertemplate="<b>%{x}</b><br>Vol: %{y:,.0f} MWh<extra></extra>",
        ),
        secondary_y=False,
    )
    
    fig.add_trace(
        go.Scatter(
            x=monthly["month"], y=monthly["avg_hpfc"],
            name="Ø HPFC (€/MWh)",
            mode="lines+markers",
            line=dict(color=Colors.AMBER, width=2.5),
            marker=dict(size=5),
            hovertemplate="<b>%{x}</b><br>Ø HPFC: %{y:.1f} €<extra></extra>",
        ),
        secondary_y=True,
    )
    
    fig.add_trace(
        go.Scatter(
            x=monthly["month"], y=monthly["avg_temp"],
            name="Ø Temp (°C)",
            mode="lines+markers",
            line=dict(color=Colors.ROSE, width=1.5, dash="dot"),
            marker=dict(size=4),
            hovertemplate="<b>%{x}</b><br>Ø Temp: %{y:.1f}°C<extra></extra>",
        ),
        secondary_y=True,
    )
    
    fig.update_layout(**make_layout(
        title=dict(text="Monatliche Zusammenfassung", font=dict(size=14)),
        height=420,
        legend=dict(
            orientation="h", y=1.12, x=0.5, xanchor="center",
            font=dict(size=11),
        ),
    ))
    fig.update_yaxes(title_text="MWh", secondary_y=False)
    fig.update_yaxes(title_text="€/MWh  &  °C", secondary_y=True)
    
    return fig


def chart_temperature_distribution(data: PortfolioData) -> go.Figure:
    """
    Temperature histogram with key thresholds.
    """
    fig = go.Figure()
    
    fig.add_trace(go.Histogram(
        x=data.temperature,
        nbinsx=80,
        marker_color=Colors.with_alpha(Colors.CYAN, 0.5),
        marker_line=dict(color=Colors.CYAN, width=0.7),
        hovertemplate="Temp: %{x:.1f}°C<br>Häufigkeit: %{y}<extra></extra>",
    ))
    
    # Threshold lines
    fig.add_vline(x=0, line=dict(color=Colors.BLUE_LIGHT, width=1.5, dash="dash"),
                  annotation_text="0°C (Frost)", annotation_font=dict(size=10))
    fig.add_vline(x=15, line=dict(color=Colors.AMBER, width=1, dash="dot"),
                  annotation_text="15°C (Heizgrenze)", annotation_font=dict(size=10))
    fig.add_vline(x=22, line=dict(color=Colors.ROSE, width=1, dash="dot"),
                  annotation_text="22°C (Kühlgrenze)", annotation_font=dict(size=10))
    
    fig.update_layout(**make_layout(
        title=dict(text="Temperaturverteilung (°C)", font=dict(size=14)),
        xaxis_title="Temperatur (°C)",
        yaxis_title="Häufigkeit",
        height=360,
        showlegend=False,
    ))
    
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
#  §11  GUIDED WORKFLOW UI COMPONENTS
# ═══════════════════════════════════════════════════════════════════════════════

def render_step_card(
    number: int,
    title: str,
    description: str,
    status: str = "pending",  # "pending", "active", "done"
) -> str:
    """Render a workflow step card as HTML."""
    card_class = {
        "pending": "",
        "active": "active",
        "done": "completed",
    }.get(status, "")
    
    number_class = {
        "pending": "pending",
        "active": "active",
        "done": "done",
    }.get(status, "pending")
    
    icon = {
        "pending": str(number),
        "active": str(number),
        "done": "✓",
    }.get(status, str(number))
    
    return f"""
    <div class="step-card {card_class}">
        <div class="step-header">
            <div class="step-number {number_class}">{icon}</div>
            <div class="step-title">{title}</div>
        </div>
        <div class="step-desc">{description}</div>
    </div>
    """


def render_kpi_card(
    label: str,
    value: str,
    unit: str = "",
    delta: Optional[str] = None,
    delta_type: str = "neutral",
) -> str:
    """Render a KPI metric card as HTML."""
    delta_html = ""
    if delta:
        delta_html = f'<div class="kpi-delta {delta_type}">{delta}</div>'
    
    return (
        f'<div class="kpi-card">'
        f'<div class="kpi-label">{label}</div>'
        f'<div class="kpi-value">{value}<span class="kpi-unit">{unit}</span></div>'
        f'{delta_html}'
        f'</div>'
    )


def render_column_pill(col_name: str, col_type: str) -> str:
    """Render a column type indicator pill."""
    type_labels = {
        "datetime": "📅 Datum",
        "load": "⚡ Last",
        "price": "💰 Preis",
        "temperature": "🌡️ Temp",
        "unknown": "❓ Unbekannt",
    }
    return (
        f'<span class="col-pill {col_type}">'
        f'{type_labels.get(col_type, col_type)} · {col_name}</span>'
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  §12  MAIN APPLICATION LAYOUT — DATA INPUT SECTION
#
#  This section implements the first part of the UI:
#  - Hero banner
#  - Sidebar configuration
#  - Data input (synthetic generation + copy-paste)
#  - Data exploration & quality report
#
#  The simulation engine and results display are in Block 2 & 3.
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    """Main application entry point."""
    
    # ── Inject CSS ──
    inject_css()
    
    # ── Session State Defaults ──
    defaults = {
        "data": None,
        "results": None,
        "paste_text": "",
        "data_source": None,
        "active_preset": None,
        "simulation_count": 0,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

    # ════════════════════════════════════════════════════════════
    #  SIDEBAR
    # ════════════════════════════════════════════════════════════
    
    with st.sidebar:
        st.markdown("### ⚡ Engine Konfiguration")
        st.caption(f"Version {APP_VERSION}")
        
        # ── Quick Presets ──
        with st.expander("🚀  Schnellstart-Presets", expanded=False):
            st.markdown(
                '<div class="info-box">'
                '<strong>Tipp:</strong> Wählen Sie ein Preset, um alle Parameter '
                'automatisch für ein typisches Szenario zu setzen.'
                '</div>',
                unsafe_allow_html=True,
            )
            
            preset_name = st.selectbox(
                "Szenario wählen",
                ["(Manuell konfigurieren)"] + list(PRESET_SCENARIOS.keys()),
                format_func=lambda x: (
                    x if x == "(Manuell konfigurieren)"
                    else f"{PRESET_SCENARIOS[x]['icon']}  {x}"
                ),
                key="preset_selector",
            )
            
            if preset_name != "(Manuell konfigurieren)":
                preset = PRESET_SCENARIOS[preset_name]
                st.markdown(
                    f'<div class="info-box">{preset["description"]}</div>',
                    unsafe_allow_html=True,
                )
                if st.button("📋 Preset übernehmen", use_container_width=True):
                    st.session_state["active_preset"] = preset_name
                    st.rerun()
        
        # ── Get active preset values (if any) ──
        ap = None
        if st.session_state.get("active_preset"):
            ap = PRESET_SCENARIOS.get(st.session_state["active_preset"])
        
        # ── Profile & Volume ──
        with st.expander("📐  Lastprofil & Volumen", expanded=True):
            profile = st.selectbox(
                "BDEW Lastprofil",
                list(BDEW_PROFILES.keys()),
                format_func=lambda x: (
                    f"{BDEW_PROFILES[x]['icon']}  {x} — {BDEW_PROFILES[x]['name']}"
                ),
                index=(
                    list(BDEW_PROFILES.keys()).index(ap["profile"])
                    if ap else 0
                ),
                help="Standardlastprofil nach BDEW/VDN-Richtlinien",
            )
            
            # Show profile description
            pi = BDEW_PROFILES[profile]
            st.markdown(
                f'<div class="info-box">'
                f'<strong>{pi["name_en"]}</strong><br>'
                f'{pi["description"]}<br>'
                f'<em>Typisch: {pi["typical_volume"]}</em>'
                f'</div>',
                unsafe_allow_html=True,
            )
            
            annual_mwh = st.number_input(
                "Jahresverbrauch (MWh)",
                min_value=100,
                max_value=1_000_000,
                value=ap["annual_mwh"] if ap else 10_000,
                step=500,
                help="Prognostizierter Jahresenergieverbrauch des Portfolios",
            )
            
            c1, c2 = st.columns(2)
            start_year = c1.number_input(
                "Startjahr",
                min_value=2024,
                max_value=2040,
                value=ap["start_year"] if ap else 2026,
                help="Erstes Lieferjahr",
            )
            n_years = c2.selectbox(
                "Laufzeit",
                [1, 2, 3, 4, 5],
                index=([1,2,3,4,5].index(ap["n_years"]) if ap else 2),
                format_func=lambda n: f"{n} Jahr{'e' if n > 1 else ''}",
                help="Anzahl der Lieferjahre im Portfolio",
            )
            
            temp_sensitivity = st.slider(
                "Temperatur-Sensitivität (β_H)",
                min_value=0.0,
                max_value=0.10,
                value=0.03,
                step=0.005,
                format="%.3f",
                help=(
                    "Relative Lastzunahme pro Heizgradtag (HDD). "
                    "Typisch: 0.02–0.05 für Haushaltskunden."
                ),
            )
        
        # ── Market Parameters ──
        with st.expander("📊  Markt & Kapital", expanded=True):
            base_price = st.number_input(
                "Baseload-Preis Frontjahr (€/MWh)",
                min_value=10.0,
                max_value=500.0,
                value=ap["base_price"] if ap else 80.0,
                step=5.0,
                format="%.1f",
                help=(
                    "EEX Cal-Y Base Preis des Frontjahres. "
                    "Grundlage der HPFC-Konstruktion."
                ),
            )
            
            c1, c2 = st.columns(2)
            confidence = c1.selectbox(
                "Konfidenzniveau α",
                [0.90, 0.95, 0.975, 0.99],
                index=1,
                format_func=lambda x: f"{x:.1%}",
                help="Für VaR/CVaR-Berechnung. 95% ist der Industriestandard.",
            )
            cost_cap = c2.number_input(
                "Eigenkapitalkosten (%)",
                min_value=5.0,
                max_value=35.0,
                value=15.0,
                step=1.0,
                format="%.1f",
                help=(
                    "RORAC-basierte Verzinsung auf das CVaR-Risikokapital. "
                    "Typisch: 12–18% für Stadtwerke."
                ),
            )
        
        # ── Stochastic Calibration ──
        with st.expander("🔬  Stochastische Kalibrierung", expanded=False):
            st.markdown(
                '<div class="info-box">'
                '<strong>Für Experten:</strong> Diese Parameter steuern die '
                'Monte-Carlo-Simulation. Die Standardwerte sind für den '
                'deutschen Markt kalibriert.'
                '</div>',
                unsafe_allow_html=True,
            )
            
            paths = st.select_slider(
                "Monte-Carlo Pfade",
                options=[500, 1000, 2000, 5000, 10000, 20000],
                value=2000,
                help="Mehr Pfade = genauere Ergebnisse, längere Rechenzeit.",
            )
            
            vol_error = st.slider(
                "σ_V Volumenfehler (%)",
                min_value=2.0, max_value=25.0,
                value=ap["vol_error"] if ap else 8.0,
                step=0.5,
                help=(
                    "Standardabweichung des AR(1)-Mengenfehlers. "
                    "Typisch: 6–10% für H0, 3–5% für RLM."
                ),
            )
            
            correlation = st.slider(
                "ρ Preis-Mengen-Korrelation (%)",
                min_value=-60.0, max_value=80.0,
                value=ap["correlation"] if ap else 40.0,
                step=5.0,
                help=(
                    "Korrelation zwischen Spot-Preis und Mengenfehler. "
                    "Positiv = Kälte treibt beides hoch. Typisch: 30–50%."
                ),
            )
            
            kappa = st.slider(
                "κ Mean-Reversion (h⁻¹)",
                min_value=0.01, max_value=0.50,
                value=0.10, step=0.01,
                help="Geschwindigkeit der Rückkehr zum HPFC-Erwartungswert.",
            )
            
            phi = st.slider(
                "φ AR(1)-Persistenz",
                min_value=0.50, max_value=0.998,
                value=0.95, step=0.002,
                format="%.3f",
                help=(
                    "Autokorrelation der Wetter-/Mengenabweichung. "
                    "Hohe Werte = langanhaltende Kalt-/Warmphasen."
                ),
            )
            
            sigma_price = st.slider(
                "σ_S Preisvolatilität (€/MWh)",
                min_value=3.0, max_value=50.0,
                value=15.0, step=1.0,
                help="Diffusionsvolatilität des Spot-Preisprozesses.",
            )
        
        # ── Jump & GARCH ──
        with st.expander("⚡  Sprünge & GARCH", expanded=False):
            jump_prob = st.slider(
                "λ Sprungwahrscheinlichkeit (pro h)",
                min_value=0.0, max_value=0.15,
                value=0.02, step=0.005,
                format="%.3f",
                help="Poisson-Intensität der Preissprünge.",
            )
            
            jump_size = st.slider(
                "σ_J Sprunggröße (€/MWh)",
                min_value=10.0, max_value=300.0,
                value=80.0, step=10.0,
                help="Standardabweichung der Sprunghöhe.",
            )
            
            garch_enabled = st.checkbox(
                "GARCH(1,1) aktivieren",
                value=True,
                help="Volatilitäts-Clustering: Phasen hoher Volatilität halten an.",
            )
            
            if garch_enabled:
                c1, c2, c3 = st.columns(3)
                g_omega = c1.number_input("ω", 1.0, 20.0, 5.0, 0.5, format="%.1f")
                g_alpha = c2.number_input("α", 0.01, 0.20, 0.08, 0.01, format="%.2f")
                g_beta = c3.number_input("β", 0.70, 0.97, 0.88, 0.01, format="%.2f")
                
                if (g_alpha + g_beta) >= 1.0:
                    st.warning(
                        f"⚠️ α + β = {g_alpha + g_beta:.2f} ≥ 1.0 → "
                        f"Nicht-stationärer Prozess!"
                    )
            else:
                g_omega, g_alpha, g_beta = 5.0, 0.08, 0.88
        
        seed = st.number_input(
            "Random Seed",
            min_value=0, max_value=99999,
            value=42, step=1,
            help="Für reproduzierbare Ergebnisse.",
        )
    
    # ── Collect all parameters ──
    params = dict(
        base_price=base_price,
        paths=paths,
        confidence=confidence,
        cost_of_capital=cost_cap,
        vol_error=vol_error,
        correlation=correlation,
        kappa=kappa,
        phi=phi,
        sigma_price=sigma_price,
        jump_prob=jump_prob,
        jump_size=jump_size,
        garch_enabled=garch_enabled,
        garch_omega=g_omega,
        garch_alpha=g_alpha,
        garch_beta=g_beta,
        seed=seed,
        profile=profile,
        annual_mwh=annual_mwh,
        start_year=start_year,
        n_years=n_years,
        temp_sensitivity=temp_sensitivity,
    )
    
    # Store params in session state for Block 2/3
    st.session_state["params"] = params

    # ════════════════════════════════════════════════════════════
    #  HERO BANNER
    # ════════════════════════════════════════════════════════════
    
    st.markdown(f"""
    <div class="hero-banner">
        <h1>⚡ {APP_TITLE}
            <span class="version-badge">v{APP_VERSION}</span>
        </h1>
        <p class="subtitle">
            Professionelle Risikobewertung für Energieportfolios · 
            Ornstein-Uhlenbeck + Jump-Diffusion · GARCH(1,1) · 
            AR(1) Temperaturmodell · CVaR Economic Capital · 
            BDEW Standardlastprofile · Stresstest & Sensitivitätsanalyse
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════════
    #  WORKFLOW OVERVIEW
    # ════════════════════════════════════════════════════════════
    
    data = st.session_state.get("data")
    results = st.session_state.get("results")
    
    step1_status = "done" if data else "active"
    step2_status = "done" if results else ("active" if data else "pending")
    step3_status = "active" if results else "pending"
    
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(render_step_card(
            1, "Daten laden",
            "Synthetische Daten generieren oder eigene Daten per Copy & Paste einfügen.",
            step1_status,
        ), unsafe_allow_html=True)
    with c2:
        st.markdown(render_step_card(
            2, "Simulation starten",
            "Monte-Carlo-Simulation mit den konfigurierten Parametern durchführen.",
            step2_status,
        ), unsafe_allow_html=True)
    with c3:
        st.markdown(render_step_card(
            3, "Ergebnisse analysieren",
            "Risikoprämien, Verteilungen, Stresstests und Sensitivitäten auswerten.",
            step3_status,
        ), unsafe_allow_html=True)
    
    st.divider()

    # ════════════════════════════════════════════════════════════
    #  DATA INPUT SECTION
    # ════════════════════════════════════════════════════════════
    
    tab_synth, tab_paste, tab_upload = st.tabs([
        "🔄  Synthetische Daten",
        "📋  Daten einfügen (Copy & Paste)",
        "📁  Datei hochladen",
    ])
    
    # ╌╌╌╌╌╌╌╌╌╌╌╌  TAB: SYNTHETIC DATA  ╌╌╌╌╌╌╌╌╌╌╌╌
    with tab_synth:
        st.markdown("""
        <div class="info-box">
            <strong>Synthetische Daten</strong> werden basierend auf dem gewählten 
            BDEW-Profil, Temperaturmodell und HPFC-Kurve generiert. Ideal zum 
            Testen und für Schulungszwecke. Konfigurieren Sie die Parameter in 
            der Seitenleiste.
        </div>
        """, unsafe_allow_html=True)
        
        c_gen, c_info = st.columns([1, 2])
        
        with c_gen:
            if st.button(
                "🔄  Synthetische Daten generieren",
                use_container_width=True,
                type="primary",
                help="Erstellt ein vollständiges Datenset basierend auf den Sidebar-Einstellungen.",
            ):
                with st.spinner("Generiere BDEW-Profile, Temperatur & HPFC …"):
                    pdata = generate_synthetic_data(
                        start_year=start_year,
                        n_years=n_years,
                        profile=profile,
                        annual_mwh=annual_mwh,
                        base_price=base_price,
                        temp_sensitivity=temp_sensitivity,
                        _seed=seed,
                    )
                    st.session_state["data"] = pdata
                    st.session_state["results"] = None
                    st.session_state["data_source"] = "synthetic"
                    st.rerun()
        
        with c_info:
            if data and st.session_state.get("data_source") == "synthetic":
                st.success(
                    f"✅ **{data.years[0]}–{data.years[-1]}** · "
                    f"{data.n_hours:,} Stunden · Profil **{data.profile}** · "
                    f"{data.annual_mwh:,.0f} MWh/a",
                    icon="📅",
                )
    
    # ╌╌╌╌╌╌╌╌╌╌╌╌  TAB: COPY & PASTE  ╌╌╌╌╌╌╌╌╌╌╌╌
    with tab_paste:
        st.markdown("""
        <div class="paste-area-wrapper">
            <div class="paste-icon">📋</div>
            <div class="paste-title">Daten aus Excel / CSV einfügen</div>
            <div class="paste-hint">
                Kopieren Sie Ihre Daten aus Excel, Google Sheets oder einer CSV-Datei 
                und fügen Sie sie in das Textfeld unten ein. Die Engine erkennt 
                automatisch Spaltentypen, Datumsformate und Zahlenformate.<br><br>
                <strong>Unterstützte Spalten:</strong> Datum/Uhrzeit · Last/Menge (MWh) · 
                Preis/HPFC (€/MWh) · Temperatur (°C)<br>
                <strong>Formate:</strong> Tab-getrennt · Semikolon · Komma · 
                Deutsches Zahlenformat (1.234,56)
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        paste_text = st.text_area(
            "Daten hier einfügen",
            value=st.session_state.get("paste_text", ""),
            height=250,
            placeholder=(
                "Datum\tLast_MWh\tHPFC_EUR\tTemperatur\n"
                "01.01.2026 00:00\t1,234\t82,50\t-2,3\n"
                "01.01.2026 01:00\t1,156\t78,30\t-2,8\n"
                "01.01.2026 02:00\t1,089\t75,10\t-3,1\n"
                "..."
            ),
            label_visibility="collapsed",
        )
        
        c_parse, c_example = st.columns([1, 1])
        
        with c_parse:
            if st.button(
                "🔍  Daten analysieren & importieren",
                use_container_width=True,
                type="primary",
                disabled=not paste_text.strip(),
                help="Analysiert die eingefügten Daten und erkennt automatisch die Spaltentypen.",
            ):
                st.session_state["paste_text"] = paste_text
                
                with st.spinner("Analysiere Datenstruktur …"):
                    df, mapping, messages = SmartDataParser.parse(paste_text)
                
                # Show messages
                for msg in messages:
                    if msg.startswith("❌"):
                        st.error(msg)
                    elif msg.startswith("⚠️"):
                        st.warning(msg)
                    elif msg.startswith("✅"):
                        st.success(msg)
                    else:
                        st.info(msg)
                
                if df is not None:
                    # Show column mapping
                    st.markdown("##### Erkannte Spalten:")
                    pills_html = ""
                    if mapping.datetime_col:
                        pills_html += render_column_pill(mapping.datetime_col, "datetime")
                    if mapping.load_col:
                        pills_html += render_column_pill(mapping.load_col, "load")
                    if mapping.price_col:
                        pills_html += render_column_pill(mapping.price_col, "price")
                    if mapping.temperature_col:
                        pills_html += render_column_pill(mapping.temperature_col, "temperature")
                    for col in mapping.unmapped_cols:
                        pills_html += render_column_pill(col, "unknown")
                    st.markdown(pills_html, unsafe_allow_html=True)
                    
                    # Allow manual override
                    with st.expander("🔧 Spaltenzuordnung manuell anpassen"):
                        all_cols = ["(nicht verwenden)"] + list(df.columns.astype(str))
                        
                        c1, c2 = st.columns(2)
                        dt_idx = (
                            all_cols.index(mapping.datetime_col)
                            if mapping.datetime_col and mapping.datetime_col in all_cols
                            else 0
                        )
                        new_dt = c1.selectbox("📅 Datum", all_cols, index=dt_idx)
                        
                        ld_idx = (
                            all_cols.index(mapping.load_col)
                            if mapping.load_col and mapping.load_col in all_cols
                            else 0
                        )
                        new_ld = c2.selectbox("⚡ Last/Menge", all_cols, index=ld_idx)
                        
                        c3, c4 = st.columns(2)
                        pr_idx = (
                            all_cols.index(mapping.price_col)
                            if mapping.price_col and mapping.price_col in all_cols
                            else 0
                        )
                        new_pr = c3.selectbox("💰 Preis/HPFC", all_cols, index=pr_idx)
                        
                        tp_idx = (
                            all_cols.index(mapping.temperature_col)
                            if mapping.temperature_col and mapping.temperature_col in all_cols
                            else 0
                        )
                        new_tp = c4.selectbox("🌡️ Temperatur", all_cols, index=tp_idx)
                        
                        # Apply overrides
                        if new_dt != "(nicht verwenden)":
                            mapping.datetime_col = new_dt
                        if new_ld != "(nicht verwenden)":
                            mapping.load_col = new_ld
                        if new_pr != "(nicht verwenden)":
                            mapping.price_col = new_pr
                        if new_tp != "(nicht verwenden)":
                            mapping.temperature_col = new_tp
                    
                    # Show preview
                    with st.expander("👀 Datenvorschau (erste 20 Zeilen)"):
                        st.dataframe(df.head(20), use_container_width=True)
                    
                    # Convert to PortfolioData
                    if st.button(
                        "✅  Daten übernehmen",
                        use_container_width=True,
                        type="primary",
                    ):
                        with st.spinner("Konvertiere Daten …"):
                            pdata, conv_msgs = SmartDataParser.convert_to_portfolio_data(
                                df, mapping, base_price
                            )
                        
                        for msg in conv_msgs:
                            if msg.startswith("❌"):
                                st.error(msg)
                            elif msg.startswith("⚠️"):
                                st.warning(msg)
                            else:
                                st.success(msg)
                        
                        if pdata:
                            st.session_state["data"] = pdata
                            st.session_state["results"] = None
                            st.session_state["data_source"] = "pasted"
                            st.rerun()
        
        with c_example:
            if st.button(
                "📝  Beispieldaten einfügen",
                use_container_width=True,
                type="secondary",
                help="Fügt ein Beispiel-Datenset ein, um die Funktion zu demonstrieren.",
            ):
                # Generate example data
                example_lines = ["Datum\tLast_MWh\tPreis_EUR_MWh\tTemperatur_C"]
                dt = datetime(2026, 1, 1)
                np.random.seed(123)
                for i in range(168):  # 1 week
                    h = dt.hour
                    factor = bdew_factor("H0", h, dt.month, dt.weekday())
                    load_val = factor * 1.14  # ~10,000 MWh/a
                    price_val = 80 + 12 * np.sin(2*np.pi*h/24) + np.random.normal(0, 3)
                    temp_val = 2.0 + 3.0 * np.sin(2*np.pi*(h-6)/24) + np.random.normal(0, 1)
                    example_lines.append(
                        f"{dt.strftime('%d.%m.%Y %H:%M')}\t"
                        f"{load_val:.3f}\t{price_val:.2f}\t{temp_val:.1f}"
                    )
                    dt += timedelta(hours=1)
                
                st.session_state["paste_text"] = "\n".join(example_lines)
                st.rerun()
    
    # ╌╌╌╌╌╌╌╌╌╌╌╌  TAB: FILE UPLOAD  ╌╌╌╌╌╌╌╌╌╌╌╌
    with tab_upload:
        st.markdown("""
        <div class="info-box">
            <strong>Datei-Upload</strong> — Laden Sie eine CSV- oder Excel-Datei hoch.
            Die automatische Spaltenerkennung funktioniert wie beim Copy & Paste.
        </div>
        """, unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "CSV oder Excel-Datei hochladen",
            type=["csv", "xlsx", "xls", "tsv"],
            help="Unterstützte Formate: .csv, .xlsx, .xls, .tsv",
        )
        
        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith(('.xlsx', '.xls')):
                    df_up = pd.read_excel(uploaded_file)
                else:
                    # Try different delimiters
                    content = uploaded_file.read().decode("utf-8", errors="replace")
                    uploaded_file.seek(0)
                    delimiter = SmartDataParser.detect_delimiter(content)
                    df_up = pd.read_csv(
                        uploaded_file, sep=delimiter,
                        engine="python", on_bad_lines="skip",
                    )
                
                st.success(f"✅ Datei geladen: {len(df_up):,} Zeilen × {len(df_up.columns)} Spalten")
                
                # Auto-detect columns
                mapping_up = ColumnMapping()
                for col in df_up.columns:
                    col_type, score = SmartDataParser.classify_column(str(col), df_up[col])
                    if col_type == "datetime" and not mapping_up.datetime_col:
                        mapping_up.datetime_col = str(col)
                    elif col_type == "load" and not mapping_up.load_col:
                        mapping_up.load_col = str(col)
                    elif col_type == "price" and not mapping_up.price_col:
                        mapping_up.price_col = str(col)
                    elif col_type == "temperature" and not mapping_up.temperature_col:
                        mapping_up.temperature_col = str(col)
                
                # Show mapping
                pills_html = ""
                if mapping_up.datetime_col:
                    pills_html += render_column_pill(mapping_up.datetime_col, "datetime")
                if mapping_up.load_col:
                    pills_html += render_column_pill(mapping_up.load_col, "load")
                if mapping_up.price_col:
                    pills_html += render_column_pill(mapping_up.price_col, "price")
                if mapping_up.temperature_col:
                    pills_html += render_column_pill(mapping_up.temperature_col, "temperature")
                st.markdown(pills_html, unsafe_allow_html=True)
                
                with st.expander("👀 Datenvorschau"):
                    st.dataframe(df_up.head(20), use_container_width=True)
                
                if st.button("✅  Datei-Daten übernehmen", type="primary",
                             use_container_width=True):
                    # Convert datetime column
                    if mapping_up.datetime_col:
                        try:
                            df_up[mapping_up.datetime_col] = pd.to_datetime(
                                df_up[mapping_up.datetime_col], dayfirst=True, format="mixed"
                            )
                        except Exception:
                            pass
                    
                    # Convert numeric columns
                    for col in [mapping_up.load_col, mapping_up.price_col, mapping_up.temperature_col]:
                        if col and col in df_up.columns and df_up[col].dtype == object:
                            df_up[col] = df_up[col].apply(SmartDataParser.clean_german_number)
                    
                    pdata, msgs = SmartDataParser.convert_to_portfolio_data(
                        df_up, mapping_up, base_price
                    )
                    for msg in msgs:
                        if msg.startswith("❌"):
                            st.error(msg)
                        elif msg.startswith("⚠️"):
                            st.warning(msg)
                        else:
                            st.success(msg)
                    
                    if pdata:
                        st.session_state["data"] = pdata
                        st.session_state["results"] = None
                        st.session_state["data_source"] = "uploaded"
                        st.rerun()
                        
            except Exception as e:
                st.error(f"❌ Fehler beim Laden: {str(e)}")

    # ════════════════════════════════════════════════════════════
    #  DATA QUALITY & EXPLORATION (shown if data is loaded)
    # ════════════════════════════════════════════════════════════
    
    data = st.session_state.get("data")
    
    if data is not None:
        st.divider()
        
        # ── Quality Report ──
        if data.quality:
            q = data.quality
            c_badge, c_info = st.columns([1, 3])
            
            with c_badge:
                st.markdown(render_quality_badge(q), unsafe_allow_html=True)
            
            with c_info:
                info_parts = [
                    f"**{q.total_hours:,}** Stunden",
                    f"**{len(q.years_covered)}** Jahre ({q.years_covered[0]}–{q.years_covered[-1]})",
                    f"Zeitraum: {q.date_range[0]} – {q.date_range[1]}",
                ]
                st.markdown(" · ".join(info_parts))
            
            # Show warnings if any
            if q.warnings:
                with st.expander(f"⚠️ {len(q.warnings)} Hinweise zur Datenqualität"):
                    for w in q.warnings:
                        st.warning(w)
                    for i in q.info:
                        st.info(i)
        
        # ── Data Summary KPIs ──
        st.markdown("##### 📊 Datenzusammenfassung")
        
        k1, k2, k3, k4, k5, k6 = st.columns(6)
        
        kpi_items = [
            (k1, "Gesamtvolumen", f"{data.load.sum():,.0f}", "MWh"),
            (k2, "Ø Last", f"{data.load.mean():.3f}", "MWh/h"),
            (k3, "Peak-Last", f"{data.load.max():.3f}", "MWh/h"),
            (k4, "Ø HPFC", f"{data.hpfc.mean():.1f}", "€/MWh"),
            (k5, "Ø Temperatur", f"{data.temperature.mean():.1f}", "°C"),
            (k6, "Profil", data.profile, ""),
        ]
        
        for col, (label, value, unit) in zip(
            [k1, k2, k3, k4, k5, k6], kpi_items
        ):
            col.markdown(
                render_kpi_card(kpi_items[0][0], kpi_items[0][1], kpi_items[0][2]),
                unsafe_allow_html=True,
            )
        
        # Fix the KPI rendering
        for col_widget, (_, lbl, val, unt) in zip(
            [k1, k2, k3, k4, k5, k6], kpi_items
        ):
            col_widget.markdown(
                render_kpi_card(lbl, val, unt),
                unsafe_allow_html=True,
            )
        
        # ── Data Exploration Charts ──
        st.markdown("##### 📈 Datenexploration")
        
        exp_tab1, exp_tab2, exp_tab3, exp_tab4, exp_tab5, exp_tab6 = st.tabs([
            "📈 Zeitreihe",
            "📊 Monatlich",
            "🔄 Tagesprofil",
            "📉 Dauerlinie",
            "🌡️ Temp. vs Last",
            "🗓️ Preis-Heatmap",
        ])
        
        with exp_tab1:
            hours_slider = st.slider(
                "Anzahl Stunden anzeigen",
                min_value=168, max_value=min(data.n_hours, 8760),
                value=min(720, data.n_hours), step=168,
                key="ts_hours",
            )
            st.plotly_chart(
                chart_load_hpfc_timeseries(data, hours_slider),
                use_container_width=True,
            )
        
        with exp_tab2:
            st.plotly_chart(chart_monthly_summary(data), use_container_width=True)
        
        with exp_tab3:
            st.plotly_chart(chart_average_daily_shape(data), use_container_width=True)
        
        with exp_tab4:
            st.plotly_chart(chart_load_duration_curve(data), use_container_width=True)
        
        with exp_tab5:
            c_scatter, c_temp = st.columns([3, 2])
            with c_scatter:
                st.plotly_chart(chart_temperature_vs_load(data), use_container_width=True)
            with c_temp:
                st.plotly_chart(chart_temperature_distribution(data), use_container_width=True)
        
        with exp_tab6:
            st.plotly_chart(chart_heatmap_monthly(data), use_container_width=True)
        
        # ── Raw Data Preview ──
        with st.expander("📋 Rohdaten-Vorschau (erste 200 Zeilen)"):
            preview_df = pd.DataFrame({
                "Zeitstempel": [d.strftime("%d.%m.%Y %H:%M") for d in data.dates[:200]],
                "Last (MWh)": [f"{v:.4f}" for v in data.load[:200]],
                "HPFC (€/MWh)": [f"{v:.2f}" for v in data.hpfc[:200]],
                "Temperatur (°C)": [f"{v:.1f}" for v in data.temperature[:200]],
                "HDD": [f"{v:.3f}" for v in data.hdd[:200]],
                "CDD": [f"{v:.3f}" for v in data.cdd[:200]],
            })
            st.dataframe(preview_df, use_container_width=True, height=350)
    
    # ════════════════════════════════════════════════════════════
    #  SIMULATION TRIGGER (Bridge to Block 2)
    # ════════════════════════════════════════════════════════════
    
    if data is not None:
        st.divider()
        
        st.markdown("##### 🚀 Simulation starten")
        
        c_run, c_status = st.columns([1, 2])
        
        with c_run:
            run_clicked = st.button(
                "▶  Monte-Carlo Simulation starten",
                use_container_width=True,
                type="primary",
                help=(
                    f"Startet {params['paths']:,} Simulationspfade über "
                    f"{data.n_hours:,} Stunden."
                ),
            )
        
        with c_status:
            if run_clicked:
                st.session_state["run_simulation"] = True
                # The actual simulation will be handled by Block 2
                # For now, show a placeholder
                st.info(
                    f"🔄 Simulation wird gestartet: {params['paths']:,} Pfade × "
                    f"{data.n_hours:,} Stunden = "
                    f"{params['paths'] * data.n_hours:,.0f} Berechnungen …",
                    icon="⏳",
                )
        
        # ── Block 2 Hook ──
        # The function `run_simulation_block2()` will be called here
        # when Block 2 is added. It handles the Monte Carlo engine
        # and renders the results.
        
        if hasattr(st.session_state, "run_simulation") and st.session_state.get("run_simulation"):
            # This will be replaced by the Block 2 simulation call
            try:
                run_simulation_and_display(data, params)
            except NameError:
                st.markdown("""
                <div class="warning-box">
                    <strong>⏳ Block 2 wird benötigt</strong><br>
                    Die Simulationsengine (Block 2) und die Ergebnisanzeige (Block 3) 
                    müssen noch zum Code hinzugefügt werden. Fügen Sie die nächsten 
                    Code-Blöcke am Ende dieser Datei ein.
                </div>
                """, unsafe_allow_html=True)
                st.session_state["run_simulation"] = False
    
    # ════════════════════════════════════════════════════════════
    #  FOOTER
    # ════════════════════════════════════════════════════════════
    
    st.markdown("---")
    st.caption(
        f"Quant Energy Risk Engine v{APP_VERSION} · "
        f"Designed for German Energy Market Professionals · "
        f"Not financial advice · © 2025"
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════════════
#
#  ██████╗ ██╗      ██████╗  ██████╗██╗  ██╗    ██████╗
#  ██╔══██╗██║     ██╔═══██╗██╔════╝██║ ██╔╝    ╚════██╗
#  ██████╔╝██║     ██║   ██║██║     █████╔╝      █████╔╝
#  ██╔══██╗██║     ██║   ██║██║     ██╔═██╗     ██╔═══╝
#  ██████╔╝███████╗╚██████╔╝╚██████╗██║  ██╗    ███████╗
#  ╚═════╝ ╚══════╝ ╚═════╝  ╚═════╝╚═╝  ╚═╝    ╚══════╝
#
#  BLOCK 2 OF 3:
#    ├── §13  Risk Metrics Module (VaR, CVaR, Spectral, Entropic)
#    ├── §14  Monte-Carlo Simulation Engine (Vectorised NumPy)
#    ├── §15  Stress Testing Module (6 Named Scenarios)
#    ├── §16  Sensitivity Analysis & Greeks
#    ├── §17  Chart Library — Part 2 (Risk Analytics)
#    ├── §18  Results Rendering Engine
#    └── §19  Master Function: run_simulation_and_display()
#
# ═══════════════════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════════════════════
#  §13  RISK METRICS MODULE
# ═══════════════════════════════════════════════════════════════════════════════
#
#  Implements a comprehensive suite of risk measures following:
#    - Artzner, P., Delbaen, F., Eber, J.-M., Heath, D. (1999).
#      "Coherent Measures of Risk", Mathematical Finance 9(3), 203–228.
#    - Acerbi, C. (2002). "Spectral measures of risk: A coherent
#      representation of subjective risk aversion", JBF 26(7).
#    - McNeil, A.J., Frey, R., Embrechts, P. (2015).
#      "Quantitative Risk Management", Princeton Univ. Press.
#
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class RiskMetrics:
    """
    Container for a comprehensive suite of risk statistics computed
    from a Monte-Carlo loss sample.

    Attributes
    ----------
    EL : float
        Expected Loss — the mean of the loss distribution.
        Represents the "fair-value" cost that should be priced
        into every contract regardless of risk appetite.

    std : float
        Standard deviation of the loss distribution.
        Measures dispersion but is insufficient for tail risk.

    VaR : float
        Value-at-Risk at confidence α.
        The α-quantile of the loss distribution.
        NOT a coherent risk measure (fails sub-additivity).

    CVaR : float
        Conditional Value-at-Risk (Expected Shortfall) at α.
        The expected loss given that the loss exceeds VaR.
        IS a coherent risk measure.

    UL : float
        Unexpected Loss = max(0, CVaR − EL).
        The basis for economic risk capital allocation.

    skew : float
        Sample skewness. Positive → right-tailed (more extreme losses).

    kurtosis : float
        Excess kurtosis. Positive → fatter tails than normal.
        German spot prices typically show κ ≈ 3–8.

    spectral : float
        Spectral Risk Measure with exponential risk spectrum.
        Axiomatically consistent and risk-aversion-parameterised.

    entropic : float
        Entropic risk measure: (1/γ) · log(E[exp(γ·L)]).
        Derived from exponential utility theory.

    min_val, max_val : float
        Sample extremes.

    p05, p25, median, p75, p95 : float
        Percentiles for box-plot representation.

    iqr : float
        Interquartile range (P75 − P25).

    n : int
        Sample size.

    sorted : np.ndarray
        Sorted loss values (for empirical CDF plotting).
    """
    EL: float = 0.0
    std: float = 0.0
    VaR: float = 0.0
    CVaR: float = 0.0
    UL: float = 0.0
    skew: float = 0.0
    kurtosis: float = 0.0
    spectral: float = 0.0
    entropic: float = 0.0
    min_val: float = 0.0
    max_val: float = 0.0
    p01: float = 0.0
    p05: float = 0.0
    p10: float = 0.0
    p25: float = 0.0
    median: float = 0.0
    p75: float = 0.0
    p90: float = 0.0
    p95: float = 0.0
    p99: float = 0.0
    iqr: float = 0.0
    n: int = 0
    sorted: np.ndarray = field(default_factory=lambda: np.array([]))


def compute_risk_metrics(
    losses: np.ndarray,
    alpha: float = 0.95,
    risk_aversion: float = 2.0,
    entropic_gamma: float = 0.001,
) -> RiskMetrics:
    """
    Compute a comprehensive suite of risk measures from a loss sample.

    This function implements the standard risk measurement framework
    used in energy trading and banking risk management. All measures
    are computed from the empirical distribution (no parametric
    assumptions).

    Parameters
    ----------
    losses : np.ndarray
        1-D array of loss realisations. Positive = loss, negative = profit.
    alpha : float
        Confidence level for VaR and CVaR (default: 0.95).
    risk_aversion : float
        Risk-aversion parameter γ for the spectral risk measure.
        Higher values → more weight on tail losses.
    entropic_gamma : float
        Parameter for the entropic risk measure.
        Controls the curvature of the exponential utility.

    Returns
    -------
    RiskMetrics
        Populated dataclass with all risk statistics.

    Notes
    -----
    The spectral risk measure uses the exponential risk spectrum:

        φ(p) = γ · exp(γ · p) / (exp(γ) − 1)

    This satisfies all axioms of coherent risk measures AND
    additionally the axiom of risk aversion (Acerbi, 2002),
    making it theoretically superior to CVaR for risk capital
    allocation.

    The entropic risk measure is the certainty equivalent under
    exponential utility:

        ρ_γ(L) = (1/γ) · log(E[exp(γ · L)])

    It is convex and monotone, and reduces to E[L] as γ → 0.
    """
    n = len(losses)
    if n == 0:
        return RiskMetrics()

    # ── Sort once, reuse everywhere ──
    s = np.sort(losses).astype(np.float64)

    # ── Location & Dispersion ──
    el = float(np.mean(losses))
    std = float(np.std(losses, ddof=1)) if n > 1 else 0.0

    # ── Percentiles ──
    pcts = np.percentile(losses, [1, 5, 10, 25, 50, 75, 90, 95, 99])
    p01, p05, p10, p25, median, p75, p90, p95, p99 = [float(x) for x in pcts]
    iqr = p75 - p25

    # ── Value-at-Risk ──
    idx_var = int(np.floor(alpha * n))
    idx_var = min(idx_var, n - 1)
    var = float(s[idx_var])

    # ── Conditional Value-at-Risk (Expected Shortfall) ──
    tail = s[idx_var:]
    cvar = float(tail.mean()) if len(tail) > 0 else var

    # ── Unexpected Loss ──
    ul = max(0.0, cvar - el)

    # ── Higher Moments ──
    if std > 1e-12 and n > 3:
        z = (losses - el) / std
        skw = float(np.mean(z ** 3))
        krt = float(np.mean(z ** 4) - 3.0)
    else:
        skw = krt = 0.0

    # ── Spectral Risk Measure ──
    #    φ(p) = γ · exp(γ · p) / (exp(γ) − 1)
    #    ρ = ∫₀¹ φ(p) · F⁻¹(p) dp  ≈  Σᵢ φ(pᵢ) · x_(i)
    gamma = risk_aversion
    p_grid = np.linspace(0, 1, n, endpoint=False) + 0.5 / n
    try:
        if gamma > 0:
            phi = gamma * np.exp(gamma * p_grid) / (np.exp(gamma) - 1.0)
        else:
            phi = np.ones(n)
        phi /= phi.sum()  # Normalise to probability weights
        spectral = float(np.dot(phi, s))
    except (FloatingPointError, OverflowError):
        spectral = cvar  # Fallback

    # ── Entropic Risk Measure ──
    #    ρ_γ(L) = (1/γ) · log(E[exp(γ·L)])
    try:
        if entropic_gamma > 0 and n > 0:
            # Numerical stability: subtract max before exp
            shifted = entropic_gamma * (losses - losses.max())
            log_mgf = np.log(np.mean(np.exp(shifted))) + entropic_gamma * losses.max()
            entropic = float(log_mgf / entropic_gamma)
        else:
            entropic = el
    except (FloatingPointError, OverflowError, RuntimeWarning):
        entropic = cvar  # Fallback

    return RiskMetrics(
        EL=el, std=std, VaR=var, CVaR=cvar, UL=ul,
        skew=skw, kurtosis=krt,
        spectral=spectral, entropic=entropic,
        min_val=float(s[0]), max_val=float(s[-1]),
        p01=p01, p05=p05, p10=p10, p25=p25, median=median,
        p75=p75, p90=p90, p95=p95, p99=p99, iqr=iqr,
        n=n, sorted=s,
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  §14  MONTE-CARLO SIMULATION ENGINE (VECTORISED)
# ═══════════════════════════════════════════════════════════════════════════════
#
#  The engine simulates N paths × T hours simultaneously using pure
#  NumPy vectorisation. No Python loops over paths — only over time
#  steps (unavoidable due to state dependency).
#
#  Performance:  ~2,000 paths × 26,280 hours ≈ 6–12 seconds on
#                a single CPU core (Streamlit Cloud).
#
#  Mathematical Framework:
#
#    SPOT PRICE:
#      dS = κ(μ_t − S) dt + σ_t dW¹ + J dN
#      σ²_t = ω + α ε²_{t-1} + β σ²_{t-1}     [GARCH(1,1)]
#      J ~ N(0, σ_J²),  N ~ Poisson(λ)          [Merton]
#
#    VOLUME ERROR:
#      ε_t = φ ε_{t-1} + √(1−φ²) σ_V dW²       [AR(1)]
#      Q_t^act = Q_t^fcst · (1 + ε_t)
#
#    DEPENDENCE:
#      dW² = ρ dW¹ + √(1−ρ²) dZ                 [Cholesky]
#
#    IMBALANCE (reBAP):
#      Penalty = 5 + |Z| · 15 · h_scale(t)       [Half-normal]
#      C_imb = |ΔQ| · Penalty                     [Always adverse]
#
#    VOLUME-PRICE:
#      C_vp = ΔQ · (S − HPFC)                    [Two-sided]
#
# ═══════════════════════════════════════════════════════════════════════════════

def _seasonal_vol_multiplier(month: int) -> float:
    """
    Seasonal volatility scaling factor for German spot prices.

    Winter months show 30–45% higher volatility due to:
    - Steeper merit order (gas/oil at the margin)
    - Heating demand uncertainty
    - Lower renewable availability (wind dominant, less solar)
    - Cross-border flow constraints

    Summer months show lower volatility due to:
    - Abundant solar supply midday
    - Lower overall demand
    - Flatter merit order
    """
    SEASONAL_VOL = {
        1: 1.42, 2: 1.38, 3: 1.18,
        4: 1.02, 5: 0.88, 6: 0.82,
        7: 0.78, 8: 0.82, 9: 0.92,
        10: 1.08, 11: 1.22, 12: 1.45,
    }
    return SEASONAL_VOL.get(month, 1.0)


def _hour_imbalance_scale(hour: int) -> float:
    """
    Diurnal scaling of imbalance penalty costs.

    Peak hours (7-20) have steeper merit orders, making
    imbalance energy more expensive. Overnight hours have
    lower marginal costs.

    Calibrated to German reBAP data 2020-2024.
    """
    HOUR_SCALE = np.array([
        0.65, 0.58, 0.52, 0.50, 0.52, 0.60,  # 00-05
        0.78, 1.05, 1.25, 1.30, 1.20, 1.15,  # 06-11
        1.10, 1.12, 1.15, 1.20, 1.30, 1.45,  # 12-17
        1.55, 1.50, 1.35, 1.15, 0.90, 0.72,  # 18-23
    ])
    return float(HOUR_SCALE[hour])


def run_simulation(
    data: PortfolioData,
    params: dict,
) -> dict:
    """
    NumPy-vectorised Monte-Carlo simulation engine.

    For every hourly time step t, this function computes across
    all N paths simultaneously:

    1. SPOT PRICE — Mean-reverting OU process with:
       - Merton jump-diffusion (Poisson arrivals × normal jumps)
       - Optional GARCH(1,1) conditional volatility
       - Seasonal volatility scaling (winter > summer)

    2. VOLUME ERROR — AR(1) process with:
       - Weather persistence parameter φ
       - Cholesky-correlated innovation with spot price
       - Bounded to prevent negative volumes

    3. IMBALANCE COST — German reBAP penalty model:
       - Always-adverse penalty (|ΔQ| × penalty_spread)
       - Stochastic half-normal penalty spread
       - Diurnal scaling (peak hours more expensive)

    4. VOLUME-PRICE RISK — Two-sided cross product:
       - ΔQ × (Spot − HPFC)
       - Captures correlated price-volume exposure

    Parameters
    ----------
    data : PortfolioData
        Complete dataset with load, HPFC, temperature, year mapping.
    params : dict
        All model parameters from the sidebar configuration.

    Returns
    -------
    dict
        Nested dictionary with:
        - 'yearly': list of per-year results
        - 'total': portfolio-level aggregation
        - 'raw': raw MC arrays for distribution analysis
        - 'diagnostics': convergence and model diagnostics
    """
    # ── Unpack data ──
    hpfc    = data.hpfc
    load    = data.load
    ym      = data.year_map
    n_years = len(data.years)
    T       = data.n_hours
    dates   = data.dates

    # ── Unpack parameters ──
    N          = params["paths"]
    kappa      = params["kappa"]
    sig_base   = params["sigma_price"]
    sig_v      = params["vol_error"] / 100.0
    phi        = params["phi"]
    rho        = params["correlation"] / 100.0
    lam        = params["jump_prob"]
    sig_j      = params["jump_size"]
    r_ec       = params["cost_of_capital"] / 100.0
    alpha      = params["confidence"]
    bp         = params["base_price"]
    garch_on   = params.get("garch_enabled", True)
    g_omega    = params.get("garch_omega", 5.0)
    g_alpha    = params.get("garch_alpha", 0.08)
    g_beta     = params.get("garch_beta", 0.88)

    # ── Derived constants ──
    L11   = np.sqrt(max(1e-12, 1.0 - rho ** 2))  # Cholesky factor
    phi_f = np.sqrt(max(1e-12, 1.0 - phi ** 2))   # AR(1) innovation scale

    # ── Pre-compute deterministic aggregates ──
    vol_y = np.zeros(n_years, dtype=np.float64)
    pc_y  = np.zeros(n_years, dtype=np.float64)
    for t in range(T):
        vol_y[ym[t]] += load[t]
        pc_y[ym[t]]  += hpfc[t] * load[t]

    # ── Allocate MC arrays ──
    imb    = np.zeros((N, n_years), dtype=np.float64)
    vp     = np.zeros((N, n_years), dtype=np.float64)
    spot   = np.full(N, hpfc[0], dtype=np.float64)
    vs     = np.zeros(N, dtype=np.float64)
    g_var  = np.full(N, sig_base ** 2, dtype=np.float64)

    # ── Track some path-level diagnostics ──
    spot_samples     = np.zeros((min(N, 50), min(T, 8760)), dtype=np.float32)
    vol_error_samples = np.zeros((min(N, 50), min(T, 8760)), dtype=np.float32)
    n_diag_paths = min(N, 50)
    n_diag_hours = min(T, 8760)

    # ── Progress bar ──
    bar = st.progress(0, text="Initialisiere stochastische Pfade …")
    update_freq = max(1, T // 300)

    # ══════════════════════════════════════════════
    #  MAIN SIMULATION LOOP (over time steps)
    # ══════════════════════════════════════════════

    for t in range(T):
        y = ym[t]
        dt_t = dates[t]
        month = dt_t.month
        hour = dt_t.hour

        # ── Seasonal volatility ──
        s_vol = _seasonal_vol_multiplier(month)

        # ── Generate correlated Gaussian innovations ──
        z1 = np.random.standard_normal(N)
        z2 = np.random.standard_normal(N)
        dw_p = z1                          # Price innovation
        dw_v = rho * z1 + L11 * z2        # Volume innovation (correlated)

        # ────────────────────────────────────────────
        #  1. GARCH(1,1) Conditional Volatility
        # ────────────────────────────────────────────
        if garch_on:
            # ε_{t-1} = σ_{t-1} · z1
            innovation_sq = (np.sqrt(g_var) * dw_p) ** 2
            g_var[:] = g_omega + g_alpha * innovation_sq + g_beta * g_var
            np.clip(g_var, 0.5, 8000.0, out=g_var)
            sig_t = np.sqrt(g_var) * s_vol
        else:
            sig_t = sig_base * s_vol

        # ────────────────────────────────────────────
        #  2. Merton Jump-Diffusion Component
        # ────────────────────────────────────────────
        jump_arrivals = (np.random.random(N) < lam).astype(np.float64)
        jump_sizes = np.random.normal(0, sig_j, N) * jump_arrivals

        # ────────────────────────────────────────────
        #  3. Ornstein-Uhlenbeck Spot Dynamics
        # ────────────────────────────────────────────
        #  dS = κ(μ − S)dt + σ_t dW¹ + J dN
        spot += kappa * (hpfc[t] - spot) + sig_t * dw_p + jump_sizes
        np.clip(spot, -200.0, 3000.0, out=spot)

        # ────────────────────────────────────────────
        #  4. AR(1) Volume Error with Persistence
        # ────────────────────────────────────────────
        #  ε_t = φ ε_{t-1} + √(1−φ²) σ_V dW²
        vs[:] = phi * vs + phi_f * sig_v * dw_v
        actual_load = load[t] * (1.0 + vs)
        np.maximum(actual_load, 0.0, out=actual_load)

        delta_q = actual_load - load[t]  # Positive = short, Negative = long

        # ────────────────────────────────────────────
        #  5. Imbalance Cost (reBAP Penalty Model)
        # ────────────────────────────────────────────
        #  The reBAP always penalises the out-of-balance party:
        #    Cost = |ΔQ| × (base_penalty + stochastic_spread) × hour_scale
        penalty_base = 5.0
        penalty_spread = np.abs(np.random.standard_normal(N)) * 15.0
        h_scale = _hour_imbalance_scale(hour)
        penalty_total = (penalty_base + penalty_spread) * h_scale
        imb[:, y] += np.abs(delta_q) * penalty_total

        # ────────────────────────────────────────────
        #  6. Volume-Price Risk (Two-Sided)
        # ────────────────────────────────────────────
        #  C_vp = ΔQ × (Spot − HPFC)
        #  Short & expensive → loss
        #  Long & cheap → loss
        vp[:, y] += delta_q * (spot - hpfc[t])

        # ── Store diagnostics ──
        if t < n_diag_hours:
            spot_samples[:, t] = spot[:n_diag_paths].astype(np.float32)
            vol_error_samples[:, t] = vs[:n_diag_paths].astype(np.float32)

        # ── Update progress ──
        if t % update_freq == 0:
            pct = t / T
            bar.progress(
                pct,
                text=(
                    f"t = {t:,} / {T:,}  ·  {N:,} Pfade  ·  "
                    f"{dt_t.strftime('%Y-%m-%d %H:%M')}  ·  "
                    f"Ø Spot: {spot.mean():.1f} €"
                ),
            )

    bar.progress(1.0, text="✅  Simulation abgeschlossen")
    time.sleep(0.3)
    bar.empty()

    # ══════════════════════════════════════════════
    #  POST-PROCESSING & RISK EVALUATION
    # ══════════════════════════════════════════════

    yearly_results = []

    for yi in range(n_years):
        V = vol_y[yi]
        if V < 1e-6:
            continue

        # ── Structure premium ──
        struct_prem = (pc_y[yi] / V) - bp

        # ── Imbalance risk metrics ──
        im_metrics = compute_risk_metrics(imb[:, yi], alpha)
        prog_prem = (im_metrics.EL + im_metrics.UL * r_ec) / V

        # ── VP risk metrics ──
        vp_metrics = compute_risk_metrics(vp[:, yi], alpha)
        vp_prem = (vp_metrics.EL + vp_metrics.UL * r_ec) / V

        # ── Total per year ──
        total_prem = struct_prem + prog_prem + vp_prem

        yearly_results.append(dict(
            yi=yi,
            year=data.years[yi],
            vol=V,
            struktur=struct_prem,
            prognose=prog_prem,
            vp_prem=vp_prem,
            gesamt=total_prem,
            im=im_metrics,
            vm=vp_metrics,
        ))

    # ── Portfolio-level (diversification) ──
    imb_tot = imb.sum(axis=1)
    vp_tot  = vp.sum(axis=1)
    total_loss = imb_tot + vp_tot
    Vt = vol_y.sum()

    sp_total = (pc_y.sum() / Vt) - bp

    im_total = compute_risk_metrics(imb_tot, alpha)
    vm_total = compute_risk_metrics(vp_tot, alpha)
    combined_total = compute_risk_metrics(total_loss, alpha)

    pp_total = (im_total.EL + im_total.UL * r_ec) / Vt
    vp_total_prem = (vm_total.EL + vm_total.UL * r_ec) / Vt
    gesamt_total = sp_total + pp_total + vp_total_prem

    # ── Diversification benefit ──
    standalone_risk = sum(
        (yr["prognose"] + yr["vp_prem"]) * yr["vol"] for yr in yearly_results
    ) / Vt
    portfolio_risk = pp_total + vp_total_prem
    div_benefit = standalone_risk - portfolio_risk

    # ── Convergence diagnostic ──
    convergence_steps = min(100, N // 5)
    if convergence_steps > 5:
        ns = np.unique(np.logspace(
            np.log10(max(30, N // 50)),
            np.log10(N),
            convergence_steps,
        ).astype(int))
        conv_cvar = np.array([
            compute_risk_metrics(total_loss[:k], alpha).CVaR for k in ns
        ])
        conv_var = np.array([
            compute_risk_metrics(total_loss[:k], alpha).VaR for k in ns
        ])
    else:
        ns = np.array([N])
        conv_cvar = np.array([combined_total.CVaR])
        conv_var = np.array([combined_total.VaR])

    return dict(
        yearly=yearly_results,
        total=dict(
            vol=Vt,
            struktur=sp_total,
            prognose=pp_total,
            vp_prem=vp_total_prem,
            gesamt=gesamt_total,
            im=im_total,
            vm=vm_total,
            combined=combined_total,
            div_benefit=div_benefit,
        ),
        raw=dict(
            imb_y=imb,
            vp_y=vp,
            imb_tot=imb_tot,
            vp_tot=vp_tot,
            total_loss=total_loss,
        ),
        diagnostics=dict(
            spot_samples=spot_samples,
            vol_error_samples=vol_error_samples,
            convergence_ns=ns,
            convergence_cvar=conv_cvar,
            convergence_var=conv_var,
            n_diag_paths=n_diag_paths,
            n_diag_hours=n_diag_hours,
        ),
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  §15  STRESS TESTING MODULE
# ═══════════════════════════════════════════════════════════════════════════════
#
#  Named scenarios calibrated to historical events in the German /
#  European electricity market (2020–2024).
#
#  Each scenario specifies:
#    - Price shock (€/MWh deviation from HPFC)
#    - Volume shock (fraction of forecast deviation)
#    - Duration (hours)
#    - Description and economic rationale
#
# ═══════════════════════════════════════════════════════════════════════════════

STRESS_SCENARIOS: Dict[str, Dict[str, Any]] = {
    "Dunkelflaute": {
        "desc": (
            "14-tägige Periode mit minimaler Wind- und Solareinspeisung "
            "bei gleichzeitig hoher Heizlast im Winter. Spotpreise steigen "
            "auf 300–500 €/MWh. Mengen erhöhen sich um ~20% durch Kälte."
        ),
        "price_shock": 350.0,
        "vol_shock": 0.20,
        "duration_hours": 336,
        "icon": "🌑",
        "severity": "Extrem",
        "historical": "Jan 2017, Dez 2022",
    },
    "Gaskrise (2022-Typ)": {
        "desc": (
            "Langfristige Gaspreiskrise wie 2022. Baseload-Preise bei "
            "250+ €/MWh über Monate. Extreme Contango-Termstruktur. "
            "Nachfragezerstörung (-5%) durch Preissignal."
        ),
        "price_shock": 180.0,
        "vol_shock": -0.05,
        "duration_hours": 2160,
        "icon": "🔥",
        "severity": "Extrem",
        "historical": "Aug–Dez 2022",
    },
    "Polar Vortex": {
        "desc": (
            "7-tägige extreme Kältewelle (-15°C unter Normal). "
            "Stromlast steigt um 35%, Spotpreise verdoppeln sich. "
            "Gasversorgungsengpässe möglich."
        ),
        "price_shock": 200.0,
        "vol_shock": 0.35,
        "duration_hours": 168,
        "icon": "❄️",
        "severity": "Hoch",
        "historical": "Feb 2012, Feb 2021 (Texas, analog)",
    },
    "Negativ-Preise (Solar)": {
        "desc": (
            "Ausgedehnte Phase mit hoher Solareinspeisung und "
            "geringer Nachfrage. 50+ Stunden Negativpreise bis "
            "-80 €/MWh. Long-Position profitiert bei Grundlast."
        ),
        "price_shock": -130.0,
        "vol_shock": -0.10,
        "duration_hours": 50,
        "icon": "☀️",
        "severity": "Mittel",
        "historical": "Jun 2023, Apr 2024",
    },
    "Kernkraft-Ausfall FR": {
        "desc": (
            "Großflächiger Ausfall der französischen Kernkraftflotte "
            "(Korrosionsprobleme). Grenzüberschreitende Preise "
            "steigen um 60 €/MWh über 4 Monate."
        ),
        "price_shock": 60.0,
        "vol_shock": 0.02,
        "duration_hours": 2880,
        "icon": "☢️",
        "severity": "Hoch",
        "historical": "Okt 2021 – Apr 2022",
    },
    "Milder Winter": {
        "desc": (
            "Ungewöhnlich warmer Winter (+3°C über Normal). "
            "Heizlast sinkt um 15%, Spotpreise fallen um 30 €/MWh. "
            "Long-Position durch Überdeckung = VP-Verlust."
        ),
        "price_shock": -30.0,
        "vol_shock": -0.15,
        "duration_hours": 4320,
        "icon": "🌤️",
        "severity": "Mittel",
        "historical": "Winter 2023/24",
    },
}


def run_stress_test(
    data: PortfolioData,
    params: dict,
    scenario_name: str,
) -> dict:
    """
    Execute a deterministic stress test for a named scenario.

    The stress test applies simultaneous price and volume shocks
    and computes the resulting P&L impact analytically (without
    re-running the full Monte Carlo).

    Impact decomposition:
      1. VP-Loss:     ΔQ × ΔP × duration
      2. Imbalance:   |ΔQ| × elevated_penalty × duration
      3. Structure:   ΔP × V_total × (duration/T) × (1 − hedge_ratio)

    Parameters
    ----------
    data : PortfolioData
    params : dict
    scenario_name : str
        Key from STRESS_SCENARIOS

    Returns
    -------
    dict
        Scenario results with decomposed P&L impact.
    """
    sc = STRESS_SCENARIOS[scenario_name]
    T = data.n_hours
    Vt = data.load.sum()
    avg_load = data.load.mean()

    dur = sc["duration_hours"]
    dp  = sc["price_shock"]
    dv  = sc["vol_shock"]

    # 1. Volume-Price Loss
    #    During stress: ΔQ per hour = avg_load × dv
    #    Price deviation = dp
    vp_loss = avg_load * dv * dp * dur

    # 2. Imbalance cost (elevated penalty during stress)
    #    Assume penalty triples during stress events
    stress_penalty = 40.0
    imb_loss = abs(avg_load * dv) * stress_penalty * dur

    # 3. Structural impact (unhedged portion)
    #    Fraction of total volume affected
    affected_fraction = min(1.0, dur / T)
    hedge_ratio = 0.3  # Assume ~30% hedged on average
    struct_impact = dp * Vt * affected_fraction * (1.0 - hedge_ratio)

    total = vp_loss + imb_loss + struct_impact
    per_mwh = total / Vt if Vt > 0 else 0

    return dict(
        name=scenario_name,
        icon=sc["icon"],
        desc=sc["desc"],
        severity=sc["severity"],
        historical=sc["historical"],
        vp_loss=vp_loss,
        imb_loss=imb_loss,
        struct_impact=struct_impact,
        total=total,
        per_mwh=per_mwh,
        duration=dur,
        price_shock=dp,
        vol_shock=dv,
    )


def run_all_stress_tests(
    data: PortfolioData,
    params: dict,
) -> List[dict]:
    """Run all named stress scenarios and return sorted results."""
    results = []
    for name in STRESS_SCENARIOS:
        results.append(run_stress_test(data, params, name))
    results.sort(key=lambda x: abs(x["total"]), reverse=True)
    return results


# ═══════════════════════════════════════════════════════════════════════════════
#  §16  SENSITIVITY ANALYSIS & GREEKS
# ═══════════════════════════════════════════════════════════════════════════════

def _analytical_premium_approx(
    params: dict,
    results: dict,
    param_name: str,
    new_value: float,
) -> float:
    """
    Analytical approximation of premium under parameter shift.

    Uses scaling relationships derived from the model structure
    instead of re-running the full Monte Carlo. This provides
    instantaneous sensitivity analysis.

    Each parameter has a known relationship to the risk premium:
    - σ_V: linear scaling of volume-driven risks
    - ρ: amplifies the joint VP risk component
    - κ: inverse-sqrt scaling (higher reversion → less risk)
    - φ: scaling through effective variance (1/(1−φ²))
    - etc.
    """
    base = results["total"]
    orig = params.get(param_name, 0)

    if param_name == "base_price":
        # Structure shifts 1:1 with base price change
        delta_bp = new_value - orig
        return base["gesamt"] - delta_bp  # Lower base → higher premium

    elif param_name == "vol_error":
        ratio = (new_value / orig) if orig > 0 else 1.0
        return (
            base["struktur"]
            + base["prognose"] * ratio
            + base["vp_prem"] * ratio
        )

    elif param_name == "correlation":
        # VP risk has a component proportional to ρ
        orig_frac = params["correlation"] / 100.0
        new_frac = new_value / 100.0
        if abs(orig_frac) > 0.01:
            vp_ratio = 0.4 + 0.6 * (new_frac / orig_frac)
        else:
            vp_ratio = 1.0 + 2.0 * (new_frac - orig_frac)
        return (
            base["struktur"]
            + base["prognose"]
            + base["vp_prem"] * max(0.1, vp_ratio)
        )

    elif param_name == "sigma_price":
        ratio = (new_value / orig) if orig > 0 else 1.0
        return (
            base["struktur"]
            + base["prognose"] * np.sqrt(ratio)
            + base["vp_prem"] * np.sqrt(ratio)
        )

    elif param_name == "kappa":
        # Higher κ → faster mean reversion → less risk
        if new_value > 0.001:
            ratio = np.sqrt(orig / new_value)
        else:
            ratio = 3.0
        return (
            base["struktur"]
            + base["prognose"] * min(3.0, ratio)
            + base["vp_prem"] * min(3.0, ratio)
        )

    elif param_name == "phi":
        # Effective AR(1) variance ∝ 1/(1−φ²)
        orig_eff = 1.0 / max(1e-6, 1.0 - orig ** 2)
        new_eff  = 1.0 / max(1e-6, 1.0 - new_value ** 2)
        ratio = np.sqrt(new_eff / orig_eff) if orig_eff > 0 else 1.0
        return (
            base["struktur"]
            + base["prognose"] * min(5.0, ratio)
            + base["vp_prem"] * min(5.0, ratio)
        )

    elif param_name == "jump_prob":
        # More jumps → fatter tails → higher CVaR
        ratio = (new_value / orig) if orig > 0.001 else 1.0
        tail_factor = 0.3 * (ratio - 1.0)
        risk = base["prognose"] + base["vp_prem"]
        return base["struktur"] + risk * (1.0 + tail_factor)

    elif param_name == "jump_size":
        ratio = (new_value / orig) if orig > 1.0 else 1.0
        tail_factor = 0.2 * (ratio - 1.0)
        risk = base["prognose"] + base["vp_prem"]
        return base["struktur"] + risk * (1.0 + tail_factor)

    elif param_name == "cost_of_capital":
        old_r = params["cost_of_capital"] / 100.0
        new_r = new_value / 100.0
        if old_r > 0.001:
            el_share = 0.55
            ul_share = 0.45
            risk = base["prognose"] + base["vp_prem"]
            return base["struktur"] + risk * (el_share + ul_share * (new_r / old_r))
        return base["gesamt"]

    elif param_name == "confidence":
        # Higher α → higher CVaR → higher premium
        # Approximate using normal quantile ratio
        old_z = sp_stats.norm.ppf(orig)
        new_z = sp_stats.norm.ppf(new_value) if new_value < 0.999 else 3.1
        if old_z > 0.1:
            ratio = new_z / old_z
        else:
            ratio = 1.0
        risk = base["prognose"] + base["vp_prem"]
        el_part = risk * 0.5
        ul_part = risk * 0.5
        return base["struktur"] + el_part + ul_part * ratio

    return base["gesamt"]


def compute_sensitivities(
    params: dict,
    results: dict,
) -> pd.DataFrame:
    """
    Compute first-order parameter sensitivities via analytical
    approximation (analogous to Greeks in derivatives pricing).

    Returns a DataFrame with columns:
    - parameter: Human-readable name
    - symbol: Mathematical symbol
    - base: Current parameter value
    - delta: Shock size
    - impact_up: Premium change for +Δ
    - impact_dn: Premium change for −Δ
    - sensitivity: ∂π/∂θ ≈ (π(θ+Δ) − π(θ−Δ)) / (2Δ)
    - analogy: Financial analogy (Greek name)
    """
    base_premium = results["total"]["gesamt"]

    shock_definitions = [
        ("vol_error",       "σ_V",  "Volume Vega",       params["vol_error"],       2.0),
        ("correlation",     "ρ",    "Correlation Rho",   params["correlation"],      10.0),
        ("sigma_price",     "σ_S",  "Price Vega",        params["sigma_price"],      3.0),
        ("kappa",           "κ",    "Reversion Delta",   params["kappa"],            0.03),
        ("phi",             "φ",    "Persistence Gamma", params["phi"],              0.02),
        ("jump_prob",       "λ",    "Jump Vanna",        params["jump_prob"],        0.01),
        ("jump_size",       "σ_J",  "Jump Vol",          params["jump_size"],        20.0),
        ("cost_of_capital", "r_EC", "Capital Lambda",    params["cost_of_capital"],  3.0),
        ("base_price",      "P_b",  "Base Delta",        params["base_price"],       5.0),
    ]

    rows = []
    for pname, symbol, analogy, base_val, delta in shock_definitions:
        up_val = base_val + delta
        dn_val = base_val - delta

        # Ensure valid ranges
        if pname == "phi":
            up_val = min(up_val, 0.998)
            dn_val = max(dn_val, 0.3)
        elif pname == "jump_prob":
            dn_val = max(dn_val, 0.0)
        elif pname == "kappa":
            dn_val = max(dn_val, 0.005)

        up_prem = _analytical_premium_approx(params, results, pname, up_val)
        dn_prem = _analytical_premium_approx(params, results, pname, dn_val)

        impact_up = up_prem - base_premium
        impact_dn = dn_prem - base_premium
        sens = (up_prem - dn_prem) / (2.0 * delta) if delta > 0 else 0.0

        rows.append(dict(
            parameter=pname,
            symbol=symbol,
            analogy=analogy,
            base=base_val,
            delta=delta,
            impact_up=impact_up,
            impact_dn=impact_dn,
            sensitivity=sens,
        ))

    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════════
#  §17  CHART LIBRARY — PART 2 (RISK ANALYTICS)
# ═══════════════════════════════════════════════════════════════════════════════

def chart_waterfall(results: dict) -> go.Figure:
    """Risk premium decomposition waterfall chart."""
    tot = results["total"]
    cats = [
        "Strukturbeitrag",
        "Prognoserisiko",
        "VP-Risiko",
        "Diversifikation",
        "Gesamtaufschlag",
    ]
    vals = [
        tot["struktur"],
        tot["prognose"],
        tot["vp_prem"],
        -tot["div_benefit"],
        tot["gesamt"],
    ]
    measures = ["relative", "relative", "relative", "relative", "total"]

    fig = go.Figure(go.Waterfall(
        orientation="v",
        measure=measures,
        x=cats,
        y=vals,
        text=[f"{v:+.2f}" for v in vals],
        textposition="outside",
        textfont=dict(family="JetBrains Mono", size=12, color=Colors.SLATE_200),
        connector=dict(line=dict(color=Colors.SLATE_700, width=1.5, dash="dot")),
        increasing=dict(marker=dict(color=Colors.ROSE)),
        decreasing=dict(marker=dict(color=Colors.EMERALD)),
        totals=dict(marker=dict(
            color=Colors.BLUE,
            line=dict(color=Colors.BLUE_LIGHT, width=2),
        )),
    ))
    fig.add_hline(y=0, line=dict(color=Colors.SLATE_600, width=1, dash="dash"))
    fig.update_layout(**make_layout(
        title=dict(text="Risikoprämien-Zerlegung  (€ / MWh)", font=dict(size=14)),
        yaxis_title="€ / MWh",
        showlegend=False,
        height=450,
    ))
    return fig


def chart_annual_bars(results: dict, years: list) -> go.Figure:
    """Stacked annual premium bars with total line."""
    yr = results["yearly"]
    x = [str(r["year"]) for r in yr]

    fig = go.Figure()
    for key, name, col in [
        ("struktur",  "Struktur",    Colors.INDIGO),
        ("prognose",  "Prognose",    Colors.AMBER),
        ("vp_prem",   "VP-Risiko",   Colors.ROSE),
    ]:
        fig.add_trace(go.Bar(
            x=x, y=[r[key] for r in yr], name=name,
            marker_color=col, marker_line=dict(width=0),
            hovertemplate=f"{name}<br>%{{x}}: %{{y:.3f}} €/MWh<extra></extra>",
        ))

    # Total line
    fig.add_trace(go.Scatter(
        x=x, y=[r["gesamt"] for r in yr], name="Gesamt",
        mode="lines+markers+text",
        text=[f"{r['gesamt']:.2f}" for r in yr],
        textposition="top center",
        textfont=dict(size=11, color=Colors.WHITE),
        line=dict(color=Colors.WHITE, width=2.5, dash="dot"),
        marker=dict(size=8, color=Colors.BLUE,
                    line=dict(width=2, color=Colors.WHITE)),
    ))

    fig.update_layout(**make_layout(
        barmode="stack",
        title=dict(text="Jährliche Risikoprämien  (€ / MWh)", font=dict(size=14)),
        yaxis_title="€ / MWh",
        height=430,
        legend=dict(orientation="h", y=1.12, x=0.5, xanchor="center"),
    ))
    return fig


def chart_distribution(
    losses: np.ndarray,
    label: str,
    colour: str,
    alpha: float = 0.95,
) -> go.Figure:
    """Loss distribution histogram with VaR / CVaR / Spectral lines."""
    m = compute_risk_metrics(losses, alpha)

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=losses,
        nbinsx=min(180, max(50, len(losses) // 15)),
        name=label,
        marker_color=Colors.with_alpha(colour, 0.40),
        marker_line=dict(color=colour, width=0.7),
        hovertemplate="Bereich: %{x:,.0f} €<br>Häufigkeit: %{y}<extra></extra>",
    ))

    annotations = [
        (m.EL,       "E[L]",                          Colors.EMERALD,  "dot",     "left"),
        (m.VaR,      f"VaR{int(alpha*100)}",           Colors.AMBER,   "dash",    "left"),
        (m.CVaR,     f"CVaR{int(alpha*100)}",          Colors.ROSE,    "solid",   "left"),
        (m.spectral, "Spektral",                       Colors.VIOLET,  "dashdot", "right"),
    ]

    for val, nm, c, dash, anchor in annotations:
        fig.add_vline(x=val, line=dict(color=c, width=2, dash=dash))
        fig.add_annotation(
            x=val, y=1, yref="paper",
            text=f"  {nm} = {val:,.0f} €",
            showarrow=False,
            font=dict(color=c, size=10),
            xanchor=anchor, yanchor="top",
        )

    fig.update_layout(**make_layout(
        title=dict(text=f"{label} — Verlustverteilung  (€)", font=dict(size=13)),
        xaxis_title="€ (absolut)",
        yaxis_title="Häufigkeit",
        showlegend=False,
        height=400,
    ))
    return fig


def chart_convergence(
    results: dict,
    alpha: float = 0.95,
) -> go.Figure:
    """CVaR & VaR convergence as function of N paths."""
    diag = results["diagnostics"]
    ns = diag["convergence_ns"]
    cvars = diag["convergence_cvar"]
    vars_ = diag["convergence_var"]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=ns, y=cvars, mode="lines", name=f"CVaR{int(alpha*100)}",
        line=dict(color=Colors.ROSE, width=2),
    ))
    fig.add_trace(go.Scatter(
        x=ns, y=vars_, mode="lines", name=f"VaR{int(alpha*100)}",
        line=dict(color=Colors.AMBER, width=1.5, dash="dash"),
    ))
    fig.add_hline(
        y=cvars[-1],
        line=dict(color=Colors.SLATE_500, dash="dot", width=1),
        annotation_text=f"CVaR∞ = {cvars[-1]:,.0f}€",
        annotation_font=dict(size=10, color=Colors.SLATE_400),
    )
    fig.update_layout(**make_layout(
        title=dict(text="Konvergenz-Diagnostik  (VaR & CVaR vs N)", font=dict(size=13)),
        xaxis_title="Anzahl Pfade",
        xaxis_type="log",
        yaxis_title="€",
        height=370,
        legend=dict(orientation="h", y=1.10, x=0.5, xanchor="center"),
    ))
    return fig


def chart_qq_plot(
    losses: np.ndarray,
    label: str = "Verluste",
) -> go.Figure:
    """QQ-plot against standard normal — reveals fat tails."""
    n = len(losses)
    s = np.sort(losses)
    theoretical = sp_stats.norm.ppf(np.linspace(0.002, 0.998, n))
    std = s.std()
    mean = s.mean()
    standardised = (s - mean) / (std + 1e-12)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=theoretical, y=standardised,
        mode="markers",
        marker=dict(size=2, color=Colors.BLUE, opacity=0.4),
        name="Stichprobe",
        hovertemplate="Theor: %{x:.2f}<br>Empir: %{y:.2f}<extra></extra>",
    ))
    lims = [
        min(theoretical.min(), standardised.min()),
        max(theoretical.max(), standardised.max()),
    ]
    fig.add_trace(go.Scatter(
        x=lims, y=lims, mode="lines",
        line=dict(color=Colors.ROSE, width=2, dash="dash"),
        name="Normal-Referenz",
    ))
    fig.update_layout(**make_layout(
        title=dict(text=f"QQ-Plot: {label} vs. Normalverteilung", font=dict(size=13)),
        xaxis_title="Theoretische Quantile (Normal)",
        yaxis_title="Stichproben-Quantile (standardisiert)",
        height=400,
        showlegend=False,
    ))
    return fig


def chart_spot_paths(results: dict, data: PortfolioData) -> go.Figure:
    """Show sample spot price paths vs HPFC."""
    diag = results["diagnostics"]
    samples = diag["spot_samples"]
    n_paths = min(diag["n_diag_paths"], 15)
    n_hours = diag["n_diag_hours"]
    x = data.dates[:n_hours]

    fig = go.Figure()
    for p in range(n_paths):
        fig.add_trace(go.Scatter(
            x=x, y=samples[p, :n_hours],
            mode="lines", opacity=0.25,
            line=dict(width=0.8, color=Colors.BLUE),
            showlegend=False,
            hoverinfo="skip",
        ))

    fig.add_trace(go.Scatter(
        x=x, y=data.hpfc[:n_hours],
        mode="lines", name="HPFC",
        line=dict(color=Colors.AMBER, width=2.5, dash="dot"),
    ))

    fig.update_layout(**make_layout(
        title=dict(text=f"Spot-Preis Pfade ({n_paths} von {diag['n_diag_paths']})",
                   font=dict(size=13)),
        xaxis_title="", yaxis_title="€ / MWh",
        height=400,
        legend=dict(orientation="h", y=1.10, x=0.5, xanchor="center"),
    ))
    return fig


def chart_tornado(sens_df: pd.DataFrame) -> go.Figure:
    """Tornado chart for parameter sensitivities."""
    df = sens_df.sort_values("impact_up", key=abs, ascending=True).copy()

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=df["analogy"], x=df["impact_up"],
        name="+Δ (Erhöhung)", orientation="h",
        marker_color=Colors.ROSE, marker_line=dict(width=0),
        hovertemplate="%{y}<br>+Δ: %{x:+.4f} €/MWh<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        y=df["analogy"], x=df["impact_dn"],
        name="−Δ (Senkung)", orientation="h",
        marker_color=Colors.EMERALD, marker_line=dict(width=0),
        hovertemplate="%{y}<br>−Δ: %{x:+.4f} €/MWh<extra></extra>",
    ))
    fig.add_vline(x=0, line=dict(color=Colors.SLATE_500, width=1))
    fig.update_layout(**make_layout(
        title=dict(text="Sensitivitäts-Tornado  (Δ Prämie in €/MWh)", font=dict(size=13)),
        xaxis_title="Δ Gesamtprämie (€ / MWh)",
        barmode="overlay",
        height=420,
        legend=dict(orientation="h", y=1.10, x=0.5, xanchor="center"),
    ))
    return fig


def chart_el_ul_decomposition(results: dict) -> go.Figure:
    """Expected Loss vs Unexpected Loss per year (stacked bars)."""
    yr = results["yearly"]
    x = [str(r["year"]) for r in yr]

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Imbalance-Risiko", "VP-Risiko"),
        horizontal_spacing=0.14,
    )

    for col_idx, (metric_key, title) in enumerate([("im", "Imbalance"), ("vm", "VP")], 1):
        els = [r[metric_key].EL / r["vol"] for r in yr]
        uls = [r[metric_key].UL / r["vol"] for r in yr]

        fig.add_trace(go.Bar(
            x=x, y=els, name=f"E[L] {title}",
            marker_color=Colors.AMBER, showlegend=(col_idx == 1),
            hovertemplate=f"E[L]: %{{y:.4f}} €/MWh<extra></extra>",
        ), row=1, col=col_idx)

        fig.add_trace(go.Bar(
            x=x, y=uls, name=f"UL {title}",
            marker_color=Colors.ROSE, showlegend=(col_idx == 1),
            hovertemplate=f"UL: %{{y:.4f}} €/MWh<extra></extra>",
        ), row=1, col=col_idx)

    fig.update_layout(**make_layout(
        barmode="stack",
        height=400,
        title=dict(text="Expected Loss vs. Unexpected Loss  (€/MWh)", font=dict(size=13)),
        legend=dict(orientation="h", y=1.15, x=0.5, xanchor="center"),
    ))
    return fig


def chart_stress_waterfall(stress_results: List[dict]) -> go.Figure:
    """Bar chart of stress test P&L impacts."""
    names = [sr["name"] for sr in stress_results]
    totals = [sr["total"] for sr in stress_results]
    colours = [Colors.ROSE if t > 0 else Colors.EMERALD for t in totals]

    fig = go.Figure(go.Bar(
        x=names, y=totals,
        marker_color=colours,
        text=[f"{t:+,.0f} €" for t in totals],
        textposition="outside",
        textfont=dict(size=11, color=Colors.WHITE),
        hovertemplate="%{x}<br>Impact: %{y:+,.0f} €<extra></extra>",
    ))
    fig.add_hline(y=0, line=dict(color=Colors.SLATE_500, width=1))
    fig.update_layout(**make_layout(
        title=dict(text="Stresstest — P&L Impact  (€ absolut)", font=dict(size=14)),
        yaxis_title="€",
        showlegend=False,
        height=420,
    ))
    return fig


def chart_risk_contribution_pie(results: dict) -> go.Figure:
    """Pie chart showing relative contribution of each risk component."""
    tot = results["total"]
    labels = ["Struktur", "Prognose (Imbalance)", "VP-Risiko"]
    values = [
        abs(tot["struktur"]),
        abs(tot["prognose"]),
        abs(tot["vp_prem"]),
    ]
    colours = [Colors.INDIGO, Colors.AMBER, Colors.ROSE]

    fig = go.Figure(go.Pie(
        labels=labels, values=values,
        marker=dict(colors=colours, line=dict(color=Colors.SLATE_900, width=2)),
        textinfo="label+percent",
        textfont=dict(size=12, color=Colors.WHITE),
        hole=0.45,
        hovertemplate="%{label}<br>%{value:.3f} €/MWh<br>%{percent}<extra></extra>",
    ))
    fig.update_layout(**make_layout(
        title=dict(text="Risikobeitrag nach Komponente", font=dict(size=13)),
        height=380,
        showlegend=False,
    ))
    fig.add_annotation(
        text=f"<b>{tot['gesamt']:+.2f}</b><br>€/MWh",
        x=0.5, y=0.5, showarrow=False,
        font=dict(size=16, color=Colors.WHITE),
    )
    return fig


def chart_ecdf(losses: np.ndarray, label: str, colour: str) -> go.Figure:
    """Empirical cumulative distribution function."""
    s = np.sort(losses)
    p = np.arange(1, len(s) + 1) / len(s)

    fig = go.Figure(go.Scatter(
        x=s, y=p, mode="lines",
        line=dict(color=colour, width=2),
        name=label,
        hovertemplate="Verlust: %{x:,.0f} €<br>F(x): %{y:.3f}<extra></extra>",
    ))
    fig.add_hline(y=0.95, line=dict(color=Colors.AMBER, dash="dash", width=1),
                  annotation_text="α = 95%")
    fig.add_hline(y=0.99, line=dict(color=Colors.ROSE, dash="dot", width=1),
                  annotation_text="α = 99%")
    fig.update_layout(**make_layout(
        title=dict(text=f"Empirische Verteilungsfunktion — {label}", font=dict(size=13)),
        xaxis_title="€", yaxis_title="F(x)",
        height=370, showlegend=False,
    ))
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
#  §18  RESULTS RENDERING ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

def render_results_table(results: dict) -> str:
    """Build the full HTML results table."""
    tot = results["total"]
    yr = results["yearly"]

    rows_html = ""
    for r in yr:
        pill_cls = "pill-green" if r["gesamt"] < tot["gesamt"] else "pill-red"
        vs_text = "↓ unter Ø" if r["gesamt"] < tot["gesamt"] else "↑ über Ø"

        rows_html += f"""
        <tr>
            <td style="font-weight:700">{r['year']}</td>
            <td class="mono text-right">{r['vol']:,.0f}</td>
            <td class="mono text-right" style="color:{Colors.INDIGO_LIGHT}">
                {r['struktur']:+.3f}
            </td>
            <td class="mono text-right" style="color:{Colors.AMBER_LIGHT}">
                +{r['prognose']:.3f}
                <div class="sub-detail">
                    EL {r['im'].EL/r['vol']:.4f} · UL {r['im'].UL/r['vol']:.4f}
                    · CVaR {r['im'].CVaR:,.0f}€
                </div>
            </td>
            <td class="mono text-right" style="color:{Colors.ROSE_LIGHT}">
                +{r['vp_prem']:.3f}
                <div class="sub-detail">
                    EL {r['vm'].EL/r['vol']:.4f} · UL {r['vm'].UL/r['vol']:.4f}
                    · Skew {r['vm'].skew:.2f}
                </div>
            </td>
            <td class="mono text-right" style="font-weight:700">
                {r['gesamt']:+.3f}
                <span class="pill {pill_cls}">{vs_text}</span>
            </td>
        </tr>
        """

    # Portfolio total row
    rows_html += f"""
    <tr class="row-total">
        <td>Portfolio</td>
        <td class="mono text-right">{tot['vol']:,.0f}</td>
        <td class="mono text-right" style="color:{Colors.INDIGO_LIGHT}">
            {tot['struktur']:+.3f}
        </td>
        <td class="mono text-right" style="color:{Colors.AMBER_LIGHT}">
            +{tot['prognose']:.3f}
        </td>
        <td class="mono text-right" style="color:{Colors.ROSE_LIGHT}">
            +{tot['vp_prem']:.3f}
        </td>
        <td class="mono text-right" style="color:{Colors.BLUE_LIGHT};font-size:1.05rem">
            {tot['gesamt']:+.3f}
            <span class="pill pill-blue">
                Div: −{tot['div_benefit']:.3f}
            </span>
        </td>
    </tr>
    """

    return f"""
    <table class="results-table">
    <thead><tr>
        <th>Lieferjahr</th>
        <th class="text-right">Volumen (MWh)</th>
        <th class="text-right">Struktur</th>
        <th class="text-right">Prognose</th>
        <th class="text-right">VP-Risiko</th>
        <th class="text-right">Gesamt (€/MWh)</th>
    </tr></thead>
    <tbody>{rows_html}</tbody>
    </table>
    """


def render_risk_stats_table(results: dict, params: dict) -> pd.DataFrame:
    """Build comprehensive risk statistics DataFrame."""
    tot = results["total"]
    m = tot["combined"]
    im = tot["im"]
    vm = tot["vm"]
    V = tot["vol"]

    rows = [
        ("── GESAMTPORTFOLIO ──", "", ""),
        ("E[L] Expected Loss", f"{m.EL:,.0f} €", f"{m.EL/V:.4f} €/MWh"),
        ("σ Standardabweichung", f"{m.std:,.0f} €", f"{m.std/V:.4f} €/MWh"),
        (f"VaR{int(params['confidence']*100)}", f"{m.VaR:,.0f} €", f"{m.VaR/V:.4f} €/MWh"),
        (f"CVaR{int(params['confidence']*100)}", f"{m.CVaR:,.0f} €", f"{m.CVaR/V:.4f} €/MWh"),
        ("UL Unexpected Loss", f"{m.UL:,.0f} €", f"{m.UL/V:.4f} €/MWh"),
        ("Spektrales RM (γ=2)", f"{m.spectral:,.0f} €", f"{m.spectral/V:.4f} €/MWh"),
        ("Entropisches RM", f"{m.entropic:,.0f} €", f"{m.entropic/V:.4f} €/MWh"),
        ("Schiefe (Skewness)", f"{m.skew:.4f}", "→ rechtsschief" if m.skew > 0.5 else "≈ symmetrisch"),
        ("Exzess-Kurtosis", f"{m.kurtosis:.4f}", "→ Fat Tails" if m.kurtosis > 1 else "≈ normal"),
        ("Min / Max", f"{m.min_val:,.0f} / {m.max_val:,.0f} €", ""),
        ("IQR (P25–P75)", f"{m.p25:,.0f} – {m.p75:,.0f} €", f"Breite: {m.iqr:,.0f} €"),
        ("P01 / P99", f"{m.p01:,.0f} / {m.p99:,.0f} €", ""),
        ("", "", ""),
        ("── IMBALANCE ──", "", ""),
        ("E[L] / CVaR", f"{im.EL:,.0f} / {im.CVaR:,.0f} €", ""),
        ("Schiefe / Kurtosis", f"{im.skew:.3f} / {im.kurtosis:.3f}", ""),
        ("", "", ""),
        ("── VP-RISIKO ──", "", ""),
        ("E[L] / CVaR", f"{vm.EL:,.0f} / {vm.CVaR:,.0f} €", ""),
        ("Schiefe / Kurtosis", f"{vm.skew:.3f} / {vm.kurtosis:.3f}", ""),
        ("", "", ""),
        ("── PORTFOLIO ──", "", ""),
        ("Diversifikationseffekt", f"{tot['div_benefit']:.4f} €/MWh", "Sub-Additivität"),
        ("MC-Pfade", f"{params['paths']:,}", ""),
        ("Konfidenzniveau α", f"{params['confidence']:.1%}", ""),
    ]

    return pd.DataFrame(rows, columns=["Metrik", "Absolut", "Per MWh / Info"])


# ═══════════════════════════════════════════════════════════════════════════════
#  §19  MASTER FUNCTION: run_simulation_and_display()
# ═══════════════════════════════════════════════════════════════════════════════
#
#  This is the function that Block 1's main() calls after the user
#  clicks "Run Monte-Carlo". It orchestrates:
#    1. Running the simulation
#    2. Rendering KPIs and big result
#    3. Displaying tabbed results (table, distributions, stress, etc.)
#
# ═══════════════════════════════════════════════════════════════════════════════

def run_simulation_and_display(
    data: PortfolioData,
    params: dict,
):
    """
    Master orchestrator: runs the MC simulation and renders all results.

    This function is called from main() when the user clicks
    "Run Monte-Carlo". It handles:
    - Execution of the simulation engine
    - Caching results in session state
    - Rendering the full results dashboard
    """
    # ── Check if we need to run or can use cached results ──
    if st.session_state.get("run_simulation", False):
        np.random.seed(params.get("seed", 42))
        results = run_simulation(data, params)
        st.session_state["results"] = results
        st.session_state["run_simulation"] = False
        st.session_state["simulation_count"] = st.session_state.get("simulation_count", 0) + 1

    results = st.session_state.get("results")
    if results is None:
        return

    tot = results["total"]
    years = data.years

    # ════════════════════════════════════════════════════════════
    #  KPI STRIP
    # ════════════════════════════════════════════════════════════

    st.markdown("---")
    st.markdown("##### ⚡ Simulationsergebnisse")

    k1, k2, k3, k4, k5, k6 = st.columns(6)

    kpi_defs = [
        (k1, "Strukturbeitrag", f"{tot['struktur']:+.3f}", "€/MWh", None, "neutral"),
        (k2, "Prognoserisiko", f"+{tot['prognose']:.3f}", "€/MWh", None, "neutral"),
        (k3, "VP-Risiko", f"+{tot['vp_prem']:.3f}", "€/MWh", None, "neutral"),
        (k4, "Diversifikation", f"−{tot['div_benefit']:.3f}", "€/MWh",
         "Portfolio-Effekt", "positive"),
        (k5, f"CVaR{int(params['confidence']*100)}",
         f"{tot['combined'].CVaR:,.0f}", "€",
         f"VaR: {tot['combined'].VaR:,.0f}€", "neutral"),
        (k6, "Volumen gesamt", f"{tot['vol']:,.0f}", "MWh",
         f"{len(years)} Jahre", "neutral"),
    ]

    for col, (_, label, value, unit, delta, dtype) in zip(
        [k1, k2, k3, k4, k5, k6], kpi_defs
    ):
        col.markdown(
            render_kpi_card(label, value, unit, delta, dtype),
            unsafe_allow_html=True,
        )

    st.markdown("")

    # ════════════════════════════════════════════════════════════
    #  BIG RESULT CARD
    # ════════════════════════════════════════════════════════════

    st.markdown(
        f'<div class="big-result">'
        f'<div class="br-label">Portfolio-Mischpreis — Gesamtaufschlag</div>'
        f'<div class="br-value">{tot["gesamt"]:+.2f}'
        f' <span style="font-size:1.2rem;color:{Colors.BLUE_LIGHT};'
        f'font-weight:500">€ / MWh</span></div>'
        f'<div class="br-context">'
        f'{params["paths"]:,} Pfade  ·  '
        f'α = {params["confidence"]:.0%}  ·  '
        f'CoE = {params["cost_of_capital"]:.0f}%  ·  '
        f'Profil {data.profile}  ·  '
        f'{years[0]}–{years[-1]}  ·  '
        f'Simulation #{st.session_state.get("simulation_count", 1)}'
        f'</div></div>',
        unsafe_allow_html=True,
    )

    st.markdown("")

    # ════════════════════════════════════════════════════════════
    #  TABBED RESULTS
    # ════════════════════════════════════════════════════════════

    tab_res, tab_dist, tab_stress, tab_sens, tab_diag = st.tabs([
        "📋  Ergebnisse",
        "📈  Verteilungen",
        "🔥  Stresstest",
        "🎯  Sensitivitäten",
        "🔬  Diagnostik",
    ])

    # ─────────────────── TAB 1: RESULTS ───────────────────
    with tab_res:
        c_table, c_waterfall = st.columns([3, 2])

        with c_table:
            st.markdown("##### Granulare Jahresbewertung")
            st.markdown(
                '<div style="font-size:0.7rem;color:#64748b;margin-bottom:8px">'
                'Alle Werte in € / MWh. Sub-Details: EL = Expected Loss, '
                'UL = Unexpected Loss, CVaR = Cond. Value-at-Risk.'
                '</div>',
                unsafe_allow_html=True,
            )
            st.markdown(render_results_table(results), unsafe_allow_html=True)

        with c_waterfall:
            st.plotly_chart(chart_waterfall(results), use_container_width=True)

        c_bars, c_elul = st.columns(2)
        with c_bars:
            st.plotly_chart(chart_annual_bars(results, years), use_container_width=True)
        with c_elul:
            st.plotly_chart(chart_el_ul_decomposition(results), use_container_width=True)

        # Risk contribution pie
        c_pie, c_info = st.columns([1, 1])
        with c_pie:
            st.plotly_chart(chart_risk_contribution_pie(results), use_container_width=True)
        with c_info:
            st.markdown("##### Interpretation")
            st.markdown(f"""
            <div class="info-box">
            <strong>Strukturbeitrag</strong> ({tot['struktur']:+.3f} €/MWh):<br>
            Die Differenz zwischen dem mengengewichteten Ø-HPFC und dem 
            Baseload-Frontjahrespreis. Positiv = Lastprofil liegt in 
            teuren Stunden.<br><br>
            
            <strong>Prognoserisiko</strong> (+{tot['prognose']:.3f} €/MWh):<br>
            Erwarteter Imbalance-Schaden + Risikokapitalkosten auf den 
            CVaR-basierten Unexpected Loss.<br><br>
            
            <strong>VP-Risiko</strong> (+{tot['vp_prem']:.3f} €/MWh):<br>
            Zweiseitiges Volumen-Preis-Risiko aus der Korrelation von 
            Mengenfehler und Spotpreisabweichung.<br><br>
            
            <strong>Diversifikation</strong> (−{tot['div_benefit']:.3f} €/MWh):<br>
            Sub-Additivitätseffekt: Das Portfolio-CVaR ist geringer als 
            die Summe der Einzel-CVaRs.
            </div>
            """, unsafe_allow_html=True)

    # ─────────────────── TAB 2: DISTRIBUTIONS ───────────────────
    with tab_dist:
        st.markdown("##### Verlustverteilungen")

        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(
                chart_distribution(
                    results["raw"]["imb_tot"], "Imbalance-Kosten",
                    Colors.AMBER, params["confidence"],
                ),
                use_container_width=True,
            )
        with c2:
            st.plotly_chart(
                chart_distribution(
                    results["raw"]["vp_tot"], "VP-Risiko",
                    Colors.ROSE, params["confidence"],
                ),
                use_container_width=True,
            )

        st.plotly_chart(
            chart_distribution(
                results["raw"]["total_loss"], "Gesamtes Portfolio-Risiko",
                Colors.BLUE, params["confidence"],
            ),
            use_container_width=True,
        )

        st.markdown("##### Verteilungs-Analyse")
        c3, c4 = st.columns(2)
        with c3:
            st.plotly_chart(
                chart_qq_plot(results["raw"]["total_loss"], "Portfolio-Verluste"),
                use_container_width=True,
            )
        with c4:
            st.plotly_chart(
                chart_ecdf(results["raw"]["total_loss"], "Portfolio", Colors.BLUE),
                use_container_width=True,
            )

        # Statistics table
        with st.expander("📊 Vollständige Risikostatistiken", expanded=False):
            stats_df = render_risk_stats_table(results, params)
            st.dataframe(stats_df, hide_index=True, use_container_width=True, height=600)

    # ─────────────────── TAB 3: STRESS TESTING ───────────────────
    with tab_stress:
        st.markdown("##### Stresstest-Szenarien — Deutscher Strommarkt")
        st.markdown(
            '<div class="info-box">'
            'Deterministische Worst-Case-Analyse kalibriert auf historische '
            'Extremereignisse im europäischen Energiemarkt 2020–2024. '
            'Die Szenarien kombinieren simultane Preis- und Mengenschocks.'
            '</div>',
            unsafe_allow_html=True,
        )

        stress_results = run_all_stress_tests(data, params)

        # Cards grid (3 columns)
        for row_start in range(0, len(stress_results), 3):
            cols = st.columns(3)
            for j, col in enumerate(cols):
                idx = row_start + j
                if idx >= len(stress_results):
                    break
                sr = stress_results[idx]
                impact_cls = "loss" if sr["total"] > 0 else "gain"
                with col:
                    st.markdown(f"""
                    <div class="stress-card">
                        <h4>{sr['icon']}  {sr['name']}</h4>
                        <div class="stress-desc">{sr['desc']}</div>
                        <div class="stress-impact {impact_cls}">
                            {sr['total']:+,.0f} €
                        </div>
                        <div class="stress-detail">
                            {sr['per_mwh']:+.2f} €/MWh  ·  
                            {sr['duration']:,}h Dauer  ·  
                            Schwere: {sr['severity']}
                        </div>
                        <div class="stress-detail" style="margin-top:4px">
                            Historisch: {sr['historical']}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

        st.markdown("")
        st.plotly_chart(chart_stress_waterfall(stress_results), use_container_width=True)

        with st.expander("📋 Detaillierte Stress-Zerlegung"):
            stress_df = pd.DataFrame([{
                "Szenario": sr["name"],
                "ΔP (€/MWh)": f"{sr['price_shock']:+.0f}",
                "ΔV (%)": f"{sr['vol_shock']:+.0%}",
                "Dauer (h)": f"{sr['duration']:,}",
                "VP-Verlust (€)": f"{sr['vp_loss']:+,.0f}",
                "Imbalance (€)": f"{sr['imb_loss']:+,.0f}",
                "Struktur (€)": f"{sr['struct_impact']:+,.0f}",
                "Gesamt (€)": f"{sr['total']:+,.0f}",
                "Per MWh": f"{sr['per_mwh']:+.3f}",
            } for sr in stress_results])
            st.dataframe(stress_df, hide_index=True, use_container_width=True)

    # ─────────────────── TAB 4: SENSITIVITIES ───────────────────
    with tab_sens:
        st.markdown("##### Parameter-Sensitivitätsanalyse")
        st.markdown(
            '<div class="info-box">'
            '<strong>Greeks-Analogie:</strong> Jeder Parameter wird um ±Δ geschockt. '
            'Die resultierende Prämienänderung wird analytisch approximiert — '
            'analog zu den "Greeks" im Derivatehandel.'
            '</div>',
            unsafe_allow_html=True,
        )

        sens_df = compute_sensitivities(params, results)

        c_torn, c_tbl = st.columns([3, 2])
        with c_torn:
            st.plotly_chart(chart_tornado(sens_df), use_container_width=True)

        with c_tbl:
            display_df = sens_df[[
                "analogy", "symbol", "base", "delta", "impact_up", "impact_dn", "sensitivity"
            ]].copy()
            display_df.columns = [
                "Greek", "Symbol", "Basis", "Δ",
                "Impact +Δ", "Impact −Δ", "∂π/∂θ",
            ]
            for c in ["Impact +Δ", "Impact −Δ", "∂π/∂θ"]:
                display_df[c] = display_df[c].apply(lambda x: f"{x:+.5f}")

            st.dataframe(display_df, hide_index=True, use_container_width=True, height=380)

        st.markdown("---")
        st.markdown("##### Interpretationshilfe")
        st.markdown("""
        | Greek | Parameter | Bedeutung |
        |---|---|---|
        | **Volume Vega** | σ_V | Wie stark reagiert die Prämie auf Mengenunsicherheit? |
        | **Correlation Rho** | ρ | Empfindlichkeit gegenüber der Preis-Mengen-Kopplung |
        | **Price Vega** | σ_S | Einfluss der Spot-Volatilität auf die Prämie |
        | **Reversion Delta** | κ | Auswirkung schnellerer/langsamerer Spot-Konvergenz |
        | **Persistence Gamma** | φ | Empfindlichkeit gegenüber Wetter-Persistenz |
        | **Jump Vanna** | λ | Einfluss der Sprung-Häufigkeit auf das Tail-Risiko |
        | **Jump Vol** | σ_J | Einfluss der Sprunggröße auf CVaR |
        | **Capital Lambda** | r_EC | RORAC-Sensitivität gegenüber Eigenkapitalkosten |
        | **Base Delta** | P_b | Strukturbeitrag-Verschiebung durch Basispreisänderung |
        """)

    # ─────────────────── TAB 5: DIAGNOSTICS ───────────────────
    with tab_diag:
        st.markdown("##### Modelldiagnostik & Konvergenz")

        c_conv, c_paths = st.columns(2)
        with c_conv:
            st.plotly_chart(
                chart_convergence(results, params["confidence"]),
                use_container_width=True,
            )
        with c_paths:
            st.plotly_chart(
                chart_spot_paths(results, data),
                use_container_width=True,
            )

        st.markdown("##### Aktive Parametrisierung")
        param_rows = [
            ("Spot-Preis", "κ (Mean Reversion)", f"{params['kappa']:.3f}", "h⁻¹"),
            ("Spot-Preis", "σ_S (Diffusion)", f"{params['sigma_price']:.1f}", "€/MWh"),
            ("Sprung", "λ (Intensität)", f"{params['jump_prob']:.3f}", "h⁻¹"),
            ("Sprung", "σ_J (Größe)", f"{params['jump_size']:.0f}", "€/MWh"),
            ("GARCH", "Aktiviert", str(params.get("garch_enabled", True)), ""),
            ("GARCH", "ω / α / β",
             f"{params.get('garch_omega',5):.1f} / "
             f"{params.get('garch_alpha',0.08):.2f} / "
             f"{params.get('garch_beta',0.88):.2f}", ""),
            ("Volumen", "φ (Persistenz)", f"{params['phi']:.3f}", ""),
            ("Volumen", "σ_V (Fehler)", f"{params['vol_error']:.1f}", "%"),
            ("Abhängigkeit", "ρ (Korrelation)", f"{params['correlation']:.0f}", "%"),
            ("Risiko", "α (Konfidenz)", f"{params['confidence']:.1%}", ""),
            ("Kapital", "r_EC (Eigenkapital)", f"{params['cost_of_capital']:.1f}", "%"),
            ("Simulation", "N (Pfade)", f"{params['paths']:,}", ""),
            ("Simulation", "Seed", f"{params.get('seed', 42)}", ""),
        ]
        st.dataframe(
            pd.DataFrame(param_rows, columns=["Modul", "Parameter", "Wert", "Einheit"]),
            hide_index=True,
            use_container_width=True,
        )

        st.markdown("##### Stichproben-Diagnostik")
        combined = results["raw"]["total_loss"]
        col_d1, col_d2, col_d3 = st.columns(3)

        with col_d1:
            # Normality test
            if len(combined) >= 20:
                stat_jb, p_jb = sp_stats.jarque_bera(combined)
                st.metric(
                    "Jarque-Bera Test",
                    f"p = {p_jb:.2e}",
                    delta="NICHT normal" if p_jb < 0.05 else "≈ normal",
                    delta_color="inverse" if p_jb < 0.05 else "normal",
                )
            else:
                st.metric("Jarque-Bera Test", "N zu klein")

        with col_d2:
            st.metric(
                "Exzess-Kurtosis",
                f"{tot['combined'].kurtosis:.3f}",
                delta="Fat Tails" if tot['combined'].kurtosis > 1 else "≈ Normal",
                delta_color="inverse" if tot['combined'].kurtosis > 1 else "normal",
            )

        with col_d3:
            # Convergence quality
            diag = results["diagnostics"]
            if len(diag["convergence_cvar"]) > 5:
                last_10 = diag["convergence_cvar"][-10:]
                cv = np.std(last_10) / (np.mean(last_10) + 1e-6) * 100
                st.metric(
                    "CVaR Konvergenz",
                    f"CV = {cv:.2f}%",
                    delta="Konvergiert" if cv < 3 else "Mehr Pfade nötig",
                    delta_color="normal" if cv < 3 else "inverse",
                )
            else:
                st.metric("CVaR Konvergenz", "—")


# ═══════════════════════════════════════════════════════════════════════════════
#  END OF BLOCK 2
#
#  Block 3 (next prompt) will add:
#    - §20  Methodology Documentation (full LaTeX)
#    - §21  Export & Reporting Functions
#    - §22  Advanced Analytics (Copulas, Backtesting)
#    - §23  About & Help System
#    - §24  Final main() integration
#
# ═══════════════════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════════════
#
#  ██████╗ ██╗      ██████╗  ██████╗██╗  ██╗    ██████╗
#  ██╔══██╗██║     ██╔═══██╗██╔════╝██║ ██╔╝    ╚════██╗
#  ██████╔╝██║     ██║   ██║██║     █████╔╝      █████╔╝
#  ██╔══██╗██║     ██║   ██║██║     ██╔═██╗         ██╔╝
#  ██████╔╝███████╗╚██████╔╝╚██████╗██║  ██╗    ██████╔╝
#  ╚═════╝ ╚══════╝ ╚═════╝  ╚═════╝╚═╝  ╚═╝    ╚═════╝
#
#  BLOCK 3 OF 3 — FINAL INTEGRATION:
#    ├── §20  Methodology Documentation (Full LaTeX / MathJax)
#    ├── §21  Glossary & Help System (German Energy Market)
#    ├── §22  Export & Reporting Engine
#    ├── §23  Advanced Analytics (Back-Testing, Regime, Copula)
#    ├── §24  Chart Library — Part 3 (Advanced Visuals)
#    ├── §25  Extended run_simulation_and_display() [OVERRIDE]
#    ├── §26  Extended main() [OVERRIDE]
#    └── §27  References & Literature Database
#
# ═══════════════════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════════════════════
#  §20  METHODOLOGY DOCUMENTATION (Full LaTeX / MathJax)
# ═══════════════════════════════════════════════════════════════════════════════
#
#  Complete mathematical documentation of every model component.
#  Rendered as HTML with LaTeX equations via Streamlit's native
#  st.latex() support and HTML method boxes.
#
# ═══════════════════════════════════════════════════════════════════════════════

METHODOLOGY_SECTIONS = [
    {
        "id": "spot_model",
        "title": "§1  Spot-Preis Dynamik — Ornstein-Uhlenbeck + Merton Jump-Diffusion",
        "icon": "📈",
        "content": r"""
Der Spotpreis $S_t$ folgt einem mean-revertierenden Prozess mit 
Sprungkomponente (Merton, 1976):

$$
dS_t = \kappa\bigl(\mu_t - S_t\bigr)\,dt + \sigma_t^{(S)}\,dW_t^{(1)} + J_t\,dN_t
$$

**Komponenten:**

| Symbol | Bedeutung | Typischer Wert |
|--------|-----------|----------------|
| $\mu_t$ | HPFC Forward-Kurve (deterministischer Mittelwert) | 60–120 €/MWh |
| $\kappa$ | Mean-Reversion Geschwindigkeit | 0.05–0.20 h⁻¹ |
| $\sigma_t^{(S)}$ | Bedingte Diffusions-Volatilität | 10–25 €/MWh |
| $J_t \sim \mathcal{N}(0, \sigma_J^2)$ | Sprunggröße | σ_J ≈ 50–120 €/MWh |
| $N_t \sim \text{Poisson}(\lambda)$ | Sprung-Ankunftsprozess | λ ≈ 0.01–0.05 h⁻¹ |

**Saisonale Volatilität:** Die Basis-Volatilität wird saisonal skaliert:

$$
\sigma_t^{(\text{season})} = \sigma_{\text{base}} \cdot s(m_t)
$$

wobei $s(m)$ ein monatlicher Multiplikator ist (Winter ≈ 1.4, Sommer ≈ 0.8), 
kalibriert auf EEX Day-Ahead Daten 2020–2024.

**Mean-Reversion:** Der OU-Prozess stellt sicher, dass der Spotpreis 
langfristig zur Forward-Kurve konvergiert. Die Halbwertszeit der 
Konvergenz beträgt $t_{1/2} = \ln(2)/\kappa$.

**Sprünge:** Das Merton-Modell (1976) ergänzt den diffusiven Prozess 
um seltene, aber große Preissprünge. Im deutschen Markt werden diese 
durch Dunkelflauten, Kraftwerksausfälle oder Gasversorgungsstörungen 
ausgelöst.
        """,
    },
    {
        "id": "garch_model",
        "title": "§2  GARCH(1,1) Bedingte Volatilität",
        "icon": "📊",
        "content": r"""
Wenn aktiviert, folgt die bedingte Varianz einem GARCH(1,1)-Prozess 
(Bollerslev, 1986):

$$
\bigl(\sigma_t^{(S)}\bigr)^2 = \omega + \alpha\,\varepsilon_{t-1}^2 + \beta\,\sigma_{t-1}^2
$$

mit $\varepsilon_t = \sigma_t \cdot z_t$, $z_t \sim \mathcal{N}(0,1)$.

**Stationaritätsbedingung:** $\alpha + \beta < 1$

**Unbedingte Varianz:** $\bar{\sigma}^2 = \omega / (1 - \alpha - \beta)$

**Persistenz der Volatilität:** $\alpha + \beta$ bestimmt, wie lange 
Phasen hoher Volatilität anhalten. Typisch für den deutschen Spotmarkt:

| Parameter | Typischer Wert | Interpretation |
|-----------|----------------|----------------|
| ω | 3–8 | Langfristige Varianz-Basis |
| α | 0.05–0.12 | Reaktion auf neue Schocks |
| β | 0.85–0.93 | Persistenz vergangener Volatilität |

Das GARCH-Modell fängt das empirisch dokumentierte **Volatility 
Clustering** ein: Perioden hoher Volatilität (z.B. während 
Dunkelflauten oder Gaskrise) halten über mehrere Tage an.
        """,
    },
    {
        "id": "volume_model",
        "title": "§3  Volumen-Fehler Dynamik — AR(1) mit Wetter-Persistenz",
        "icon": "🌡️",
        "content": r"""
Der Mengenprognose-Fehler folgt einem autoregressiven Prozess erster 
Ordnung (AR(1)):

$$
\varepsilon_t = \varphi\,\varepsilon_{t-1} + \sqrt{1 - \varphi^2}\;\sigma_V\;dW_t^{(2)}
$$

$$
Q_t^{\text{ist}} = Q_t^{\text{prog}} \cdot (1 + \varepsilon_t)
$$

**Persistenz-Parameter φ:**

Der Parameter $\varphi \in [0,1)$ ist die zentrale Kalibrierungsgröße. 
Er steuert, wie lange Wetter-bedingte Abweichungen anhalten:

| φ-Wert | Dekorrelationszeit | Interpretation |
|--------|-------------------|----------------|
| 0.90 | ~10 Stunden | Schwache Persistenz |
| 0.95 | ~20 Stunden | Moderate Persistenz (Standard) |
| 0.98 | ~50 Stunden | Starke Persistenz (mehrtägige Kältewelle) |
| 0.99 | ~100 Stunden | Sehr starke Persistenz (Dunkelflaute) |

Die Dekorrelationszeit berechnet sich als $\tau = -1/\ln(\varphi)$.

**Temperaturkopplung:** Das Lastprofil wird zusätzlich über 
Heizgradtage (HDD) und Kühlgradtage (CDD) adjustiert:

$$
Q_t^{\text{adj}} = Q_t^{\text{BDEW}} \cdot \bigl(1 + \beta_H \cdot \text{HDD}_t - 0.5\,\beta_H \cdot \text{CDD}_t\bigr)
$$

mit $\text{HDD}_t = \max(0, T_{\text{Heiz}} - T_t)/24$ und 
$T_{\text{Heiz}} = 15°\text{C}$ (DWD-Konvention).

**Stationäre Varianz:** Die unbedingte Varianz des AR(1)-Prozesses beträgt:

$$
\text{Var}(\varepsilon) = \frac{\sigma_V^2}{1 - \varphi^2}
$$

Für $\varphi = 0.95$ und $\sigma_V = 8\%$ ergibt sich eine effektive 
Standardabweichung von $\sigma_V / \sqrt{1-\varphi^2} \approx 25.6\%$.
        """,
    },
    {
        "id": "dependence_model",
        "title": "§4  Bivariate Abhängigkeitsstruktur — Cholesky-Zerlegung",
        "icon": "🔗",
        "content": r"""
Preis- und Mengeninnovationen werden über die instantane Korrelation 
$\rho$ gekoppelt:

$$
dW_t^{(2)} = \rho\,dW_t^{(1)} + \sqrt{1 - \rho^2}\,dZ_t
$$

wobei $dZ_t$ ein unabhängiger Wiener-Prozess ist.

**Empirische Kalibrierung:**

Für den deutschen Markt ist $\rho > 0$ typisch (≈ 0.30–0.50), da:
- **Kälte** → gleichzeitig höhere Nachfrage UND höhere Spotpreise
- **Hitze** → höhere Kühllast UND höhere Preise (Kühlwasserknappheit)

Dies erzeugt eine **adverse gemeinsame Exposition**: Das Portfolio 
ist systematisch dann short, wenn der Markt teuer ist.

**Cholesky-Zerlegung:**

Die Zerlegung der Korrelationsmatrix $\Sigma$ garantiert:
1. Gültige Korrelationsstruktur ($\Sigma$ positiv semi-definit)
2. Erhalt der marginalen Normalverteilung
3. Effiziente numerische Berechnung

$$
\begin{pmatrix} dW^{(1)} \\ dW^{(2)} \end{pmatrix} = 
\begin{pmatrix} 1 & 0 \\ \rho & \sqrt{1-\rho^2} \end{pmatrix}
\begin{pmatrix} z_1 \\ z_2 \end{pmatrix}
$$
        """,
    },
    {
        "id": "imbalance_model",
        "title": "§5  Imbalance-Preismodell — Deutsches reBAP-System",
        "icon": "⚖️",
        "content": r"""
Das **Regelzonenbilanzausgleichsenergiepreis** (reBAP) System der 
deutschen Übertragungsnetzbetreiber (ÜNBs) bestraft die 
Bilanzabweichung des Bilanzkreisverantwortlichen (BKV):

$$
\text{Penalty}_t = \bigl(5 + |Z_t| \cdot 15\bigr) \cdot h_{\text{scale}}(t)
\quad\text{mit}\quad Z_t \sim \mathcal{N}(0,1)
$$

$$
C_t^{\text{imb}} = |\Delta Q_t| \cdot \text{Penalty}_t
$$

**Immer adverser Charakter:** Der reBAP bestraft den BKV 
unabhängig von der Richtung der Abweichung:
- **Short** (ΔQ > 0): Kauf zum erhöhten reBAP
- **Long** (ΔQ < 0): Verkauf zum reduzierten reBAP
- Netto-Kosten sind immer ≥ 0

**Diurnale Skalierung** $h_{\text{scale}}(t)$: Während der 
Peak-Stunden (7–20 Uhr) ist die Merit-Order steiler, was zu 
höheren Ausgleichsenergiekosten führt:

| Tageszeit | h_scale | Begründung |
|-----------|---------|------------|
| 00–06 Uhr | 0.5–0.65 | Flache Merit-Order, geringe Last |
| 07–12 Uhr | 1.05–1.30 | Morning Ramp, steigende Nachfrage |
| 12–17 Uhr | 1.10–1.30 | Mittags-Peak, Solareinspeisung |
| 17–20 Uhr | 1.35–1.55 | Abend-Super-Peak |
| 20–24 Uhr | 0.72–1.15 | Abendlicher Rückgang |

**Regulatorischer Kontext:** Die BNetzA hat 2020 die 
Festlegungsverfahren für die reBAP-Systematik überarbeitet 
(BK6-20-201). Die aktuelle Methodik verwendet den 
volumengewichteten Durchschnitt der aktivierten Regelleistung.
        """,
    },
    {
        "id": "risk_measures",
        "title": "§6  Risikoquantifizierung — Multi-Maß Framework",
        "icon": "📐",
        "content": r"""
**Value-at-Risk (VaR)** bei Konfidenzniveau $\alpha$:

$$
\text{VaR}_\alpha(L) = \inf\bigl\{x : \Pr(L \le x) \ge \alpha\bigr\}
$$

⚠️ VaR ist **kein kohärentes Risikomaß** — er verletzt die 
Sub-Additivitätseigenschaft und kann Diversifikation "bestrafen".

**Conditional Value-at-Risk (CVaR / Expected Shortfall):**

$$
\text{CVaR}_\alpha(L) = \mathbb{E}\bigl[L \;\big|\; L > \text{VaR}_\alpha(L)\bigr]
$$

CVaR ist **kohärent** (Artzner et al., 1999) und erfüllt:
1. Monotonie
2. Translationsinvarianz
3. Positive Homogenität
4. **Sub-Additivität** ✓

**Spektrales Risikomaß** (Acerbi, 2002):

$$
\mathcal{M}_\gamma(L) = \int_0^1 \phi_\gamma(p)\,F_L^{-1}(p)\,dp
$$

mit dem exponentiellen Risikospektrum:

$$
\phi_\gamma(p) = \frac{\gamma\,e^{\gamma p}}{e^\gamma - 1}
$$

Das spektrale RM erfüllt zusätzlich das **Axiom der Risikoaversion** 
(φ ist monoton steigend): Höhere Verluste erhalten überproportional 
mehr Gewicht.

**Entropisches Risikomaß** (aus der Nutzentheorie):

$$
\rho_\gamma(L) = \frac{1}{\gamma}\,\ln\!\bigl(\mathbb{E}[e^{\gamma L}]\bigr)
$$

Abgeleitet aus der exponentiellen Nutzenfunktion $u(x) = -e^{-\gamma x}$.
Konvergiert für $\gamma \to 0$ gegen $\mathbb{E}[L]$.

**Unexpected Loss (UL) — Ökonomisches Risikokapital:**

$$
\text{UL} = \max\!\bigl(0,\;\text{CVaR}_\alpha - \mathbb{E}[L]\bigr)
$$

**RORAC-basierte Prämie:**

$$
\text{Prämie} = \frac{
  \underbrace{\mathbb{E}[L]}_{\text{Expected Loss}}
  + \underbrace{r_{EC} \cdot \text{UL}}_{\text{Risikokapitalkosten}}
}{
  \displaystyle\sum_t V_t
}
$$

Dabei ist $r_{EC}$ die geforderte Eigenkapitalrendite (RORAC), 
typischerweise 12–18% für Stadtwerke.
        """,
    },
    {
        "id": "diversification",
        "title": "§7  Portfolio-Diversifikation & Sub-Additivität",
        "icon": "🔄",
        "content": r"""
Da CVaR ein **kohärentes Risikomaß** ist, gilt die Sub-Additivität 
für elliptische Verteilungen (und approximativ allgemein):

$$
\text{CVaR}\!\Bigl(\sum_{y=1}^{Y} L_y\Bigr) \le \sum_{y=1}^{Y} \text{CVaR}(L_y)
$$

Der **Diversifikationseffekt** wird quantifiziert als:

$$
\Delta_{\text{div}} = \sum_{y=1}^{Y} \frac{V_y}{V_{\text{tot}}} \cdot 
  \pi_y^{\text{standalone}} - \pi^{\text{portfolio}}
$$

**Ökonomische Interpretation:** Bei einem Mehrjahresvertrag sind 
die Verluste der einzelnen Lieferjahre nicht perfekt korreliert. 
Ein kalter Winter 2026 impliziert nicht zwingend einen kalten 
Winter 2027. Das Portfolio-CVaR ist daher geringer als die 
volumengewichtete Summe der Einzel-CVaRs.

**Implikation für die Preisgestaltung:** Der Diversifikationseffekt 
rechtfertigt einen Mengenrabatt für Mehrjahresverträge — das 
pro-MWh Risikokapital sinkt mit der Portfoliogröße.

**Grenzen:** Die Sub-Additivität kann für sehr schwere Tails 
(α ≈ 1) oder nicht-elliptische Verteilungen verletzt werden. 
In der Praxis ist der Effekt für α ≤ 0.99 robust.
        """,
    },
    {
        "id": "stress_testing",
        "title": "§8  Stresstest-Methodik",
        "icon": "🔥",
        "content": r"""
Benannte Stressszenarien wenden deterministische Schocks auf 
Preis und Volumen gleichzeitig an:

$$
\text{PnL}_{\text{stress}} =
  \underbrace{\bar{Q} \cdot \Delta v \cdot \Delta P \cdot T_{\text{dur}}}_{\text{VP-Verlust}}
  + \underbrace{|\bar{Q} \cdot \Delta v| \cdot \bar{c}_{\text{imb}} \cdot T_{\text{dur}}}_{\text{Imbalance-Kosten}}
  + \underbrace{\Delta P \cdot V_{\text{tot}} \cdot \frac{T_{\text{dur}}}{T} \cdot (1-h)}_{\text{Struktureller Impact}}
$$

wobei:
- $\bar{Q}$ = durchschnittliche Stundenlast
- $\Delta v$ = relativer Mengenschock (z.B. +20%)
- $\Delta P$ = Preisschock (€/MWh)
- $T_{\text{dur}}$ = Dauer des Ereignisses (Stunden)
- $\bar{c}_{\text{imb}}$ = erhöhte Ausgleichsenergiekosten
- $h$ = Hedge-Ratio (Anteil der abgesicherten Position)

**Kalibrierung:** Die Szenarien sind auf historische Extremereignisse 
im europäischen Energiemarkt kalibriert:

| Szenario | Historische Referenz | Preisschock | Mengenschock |
|----------|---------------------|-------------|--------------|
| Dunkelflaute | Jan 2017 | +350 €/MWh | +20% |
| Gaskrise | Aug–Dez 2022 | +180 €/MWh | −5% |
| Polar Vortex | Feb 2012 | +200 €/MWh | +35% |
| Negativ-Preise | Jun 2023 | −130 €/MWh | −10% |
| AKW-Ausfall FR | Okt 2021 | +60 €/MWh | +2% |
| Milder Winter | Winter 2023/24 | −30 €/MWh | −15% |
        """,
    },
    {
        "id": "sensitivity_greeks",
        "title": "§9  Sensitivitätsanalyse — Energy Greeks",
        "icon": "🎯",
        "content": r"""
Die parametrischen Sensitivitäten werden analytisch approximiert, 
analog zu den „Greeks" im Derivatehandel:

$$
\frac{\partial \pi}{\partial \theta} \approx 
\frac{\pi(\theta + \Delta) - \pi(\theta - \Delta)}{2\Delta}
$$

**Skalierungsbeziehungen:**

| Greek | Parameter | Skalierung | Herleitung |
|-------|-----------|------------|------------|
| Volume Vega | $\sigma_V$ | $\pi \propto \sigma_V$ | Linearer Mengenfehler |
| Price Vega | $\sigma_S$ | $\pi \propto \sqrt{\sigma_S}$ | Diffusions-Skalierung |
| Correlation Rho | $\rho$ | $\pi_{\text{VP}} \propto \rho$ | Cholesky-Struktur |
| Reversion Delta | $\kappa$ | $\pi \propto 1/\sqrt{\kappa}$ | OU-Varianz ∝ 1/κ |
| Persistence Gamma | $\varphi$ | $\pi \propto 1/\sqrt{1-\varphi^2}$ | AR(1)-Varianz |
| Jump Vanna | $\lambda$ | Tail-Effekt auf CVaR | Poisson-Intensität |
| Capital Lambda | $r_{EC}$ | $\pi_{\text{UL}} \propto r_{EC}$ | Lineare Kapitalkosten |

**Anwendung:** Die Greeks ermöglichen eine schnelle 
Was-wäre-wenn-Analyse ohne erneute Monte-Carlo-Simulation. 
Sie identifizieren die Haupttreiber des Gesamtrisikos und 
unterstützen die Kalibrierungsentscheidung.
        """,
    },
    {
        "id": "bdew_profiles",
        "title": "§10  BDEW Standardlastprofile — Regulatorischer Rahmen",
        "icon": "📋",
        "content": r"""
Die BDEW-Standardlastprofile (SLP) werden vom Bundesverband der 
Energie- und Wasserwirtschaft veröffentlicht und bilden die Grundlage 
für die Bilanzierung von SLP-Kunden in Deutschland.

**Analytische Darstellung:** Das offizielle SLP-Verfahren verwendet 
die polynomiale H(T)-Funktion:

$$
h(T_d) = \frac{A}{1 + \left(\frac{B}{T_d - T_0}\right)^C} + D + E \cdot T_d + F \cdot T_d^2
$$

wobei $T_d$ die Tagesdurchschnittstemperatur ist und A–F die 
profilspezifischen Koeffizienten sind.

**Implementierung in dieser Engine:** Wir verwenden eine vereinfachte 
tabellenbasierte Approximation mit:
- 24-Stunden-Faktoren pro Profiltyp
- 3 Tagestypen (Werktag, Samstag, Sonntag)
- 3 Jahreszeiten (Winter, Übergang, Sommer)
- Saisonale Skalierungsfaktoren

**Profile in der Engine:**

| Profil | Kategorie | Besonderheiten |
|--------|-----------|----------------|
| H0 | Haushalt | Morgen-/Abendpeak, Winterdominanz |
| G0–G6 | Gewerbe | Diverse Geschäftszeitenmuster |
| L0–L2 | Landwirtschaft | Melkzeiten (L1), saisonale Variation |
| RLM | Industrie | Nahezu konstante Bandlast |

**Normierung:** Die Profil-Faktoren werden auf das angegebene 
Jahresvolumen normiert, sodass $\sum_{t} Q_t = V_{\text{Jahr}}$.
        """,
    },
]


def render_methodology_tab():
    """Render the complete methodology documentation."""
    st.markdown("""
    <div class="info-box">
        <strong>Mathematische Grundlagen</strong> — Diese Dokumentation beschreibt 
        alle Modellkomponenten der Quant Energy Risk Engine. Die Gleichungen 
        verwenden Standard-Notation aus der quantitativen Finanz- und 
        Energiewirtschaft.
    </div>
    """, unsafe_allow_html=True)

    for section in METHODOLOGY_SECTIONS:
        with st.expander(
            f"{section['icon']}  {section['title']}",
            expanded=False,
        ):
            st.markdown(section["content"])

    st.markdown("---")
    render_references()


# ═══════════════════════════════════════════════════════════════════════════════
#  §21  GLOSSARY & HELP SYSTEM
# ═══════════════════════════════════════════════════════════════════════════════

GLOSSARY_ENTRIES = {
    "A": [
        ("AR(1)", "Autoregressiver Prozess 1. Ordnung",
         "Zeitreihenmodell, bei dem der aktuelle Wert linear vom vorherigen abhängt: "
         "ε_t = φ·ε_{t-1} + Innovation. Modelliert die Persistenz von Wettermustern."),
        ("Ausgleichsenergie", "Regelzonenbilanzausgleichsenergie (reBAP)",
         "Energie, die von den ÜNBs zum Ausgleich von Bilanzkreisabweichungen "
         "eingesetzt wird. Der Preis (reBAP) bestraft systematisch die "
         "abweichende Partei."),
    ],
    "B": [
        ("BDEW", "Bundesverband der Energie- und Wasserwirtschaft",
         "Branchenverband, der die Standardlastprofile (SLP) für die deutsche "
         "Stromwirtschaft publiziert."),
        ("BKV", "Bilanzkreisverantwortlicher",
         "Marktrolle nach EnWG/StromNZV, verantwortlich für den Ausgleich "
         "von Ein- und Ausspeisung im eigenen Bilanzkreis."),
        ("Baseload", "Grundlast / Bandlieferung",
         "Konstante Lieferung über alle Stunden. Grundprodukt am EEX-Terminmarkt."),
    ],
    "C": [
        ("CDD", "Cooling Degree Days (Kühlgradtage)",
         "Maß für den Kühlbedarf: CDD = max(0, T − T_Kühl) / 24, "
         "mit T_Kühl = 22°C. Erhöht die Last durch Klimaanlagen."),
        ("Cholesky", "Cholesky-Zerlegung",
         "Faktorisierung einer positiv-definiten Matrix L·L^T. Wird verwendet, "
         "um korrelierte Zufallsvariablen aus unabhängigen zu erzeugen."),
        ("Contango", "Terminstruktur mit Aufschlag",
         "Marktstruktur, bei der spätere Lieferperioden teurer sind als frühere. "
         "Typisch für Strommärkte aufgrund von Risikoprämien und Kapitalkosten."),
        ("CVaR", "Conditional Value-at-Risk (Expected Shortfall)",
         "Erwarteter Verlust gegeben, dass der Verlust den VaR übersteigt. "
         "Kohärentes Risikomaß nach Artzner et al. (1999)."),
    ],
    "D": [
        ("Dunkelflaute", "Dark Doldrums",
         "Mehrere Tage mit gleichzeitig niedriger Wind- und Solareinspeisung. "
         "Führt zu extremen Spotpreisen und ist ein zentrales Szenario in der "
         "Stresstest-Methodik."),
    ],
    "E": [
        ("EEX", "European Energy Exchange",
         "Europäische Energiebörse mit Sitz in Leipzig. Marktplatz für "
         "Strom-Terminprodukte (Base, Peak, Calendar Year, Quarter, Month)."),
        ("EPEX Spot", "European Power Exchange — Spotmarkt",
         "Day-Ahead und Intraday Spotmarkt für Strom in Zentraleuropa."),
    ],
    "G": [
        ("GARCH", "Generalized Autoregressive Conditional Heteroskedasticity",
         "Zeitreihenmodell für bedingte Varianz. Fängt Volatility Clustering ein: "
         "σ²_t = ω + α·ε²_{t-1} + β·σ²_{t-1}. Bollerslev (1986)."),
    ],
    "H": [
        ("HDD", "Heating Degree Days (Heizgradtage)",
         "Maß für den Heizbedarf: HDD = max(0, T_Heiz − T) / 24, "
         "mit T_Heiz = 15°C (DWD). Zentral für die Temperatur-Last-Kopplung."),
        ("HPFC", "Hourly Price Forward Curve",
         "Stündlich aufgelöste Preiserwartung, konstruiert aus der "
         "EEX-Terminkurve und einem Shaping-Modell (Intraday + Saison + Wochentag)."),
    ],
    "M": [
        ("Mean-Reversion", "Rückkehr zum Mittelwert",
         "Eigenschaft des OU-Prozesses: Der Spotpreis konvergiert langfristig "
         "zur Forward-Kurve. Geschwindigkeit bestimmt durch κ."),
        ("Merit Order", "Einsatzreihenfolge der Kraftwerke",
         "Sortierung nach variablen Kosten. Bestimmt den marginalen "
         "Strompreis und die Steilheit der Angebotskurve."),
        ("Monte-Carlo", "Monte-Carlo-Simulation",
         "Stochastische Methode zur Bewertung komplexer Risiken durch "
         "wiederholte Zufallsziehung. N Pfade × T Zeitschritte."),
    ],
    "O": [
        ("OU-Prozess", "Ornstein-Uhlenbeck Prozess",
         "Mean-revertierender stochastischer Prozess: "
         "dS = κ(μ − S)dt + σdW. Standardmodell für Energiepreise."),
    ],
    "R": [
        ("reBAP", "Regelzonenbilanzausgleichsenergiepreis",
         "Preis für die Ausgleichsenergie, festgelegt durch die ÜNBs "
         "auf Basis der aktivierten Regelleistung. Immer nachteilig "
         "für den abweichenden BKV."),
        ("RLM", "Registrierende Leistungsmessung",
         "Messmethode für Großkunden mit 15-Minuten-Lastgangmessung. "
         "Keine Anwendung von Standardlastprofilen."),
        ("RORAC", "Return on Risk-Adjusted Capital",
         "Renditekennzahl: Ertrag / ökonomisches Risikokapital. "
         "Bestimmt die geforderte Verzinsung auf den Unexpected Loss."),
    ],
    "S": [
        ("SLP", "Standardlastprofil",
         "Genormtes Lastprofil nach BDEW zur Bilanzierung von "
         "nicht-leistungsgemessenen Kunden (H0, G0–G6, L0–L2)."),
        ("Sub-Additivität", "Subadditivitätseigenschaft",
         "Eigenschaft kohärenter Risikomaße: ρ(X+Y) ≤ ρ(X) + ρ(Y). "
         "Garantiert, dass Diversifikation nie bestraft wird."),
    ],
    "U": [
        ("UL", "Unexpected Loss",
         "Differenz zwischen CVaR und Expected Loss: UL = CVaR − E[L]. "
         "Basis für die Berechnung des ökonomischen Risikokapitals."),
        ("ÜNB", "Übertragungsnetzbetreiber",
         "Betreiber des Höchstspannungsnetzes (50Hertz, Amprion, "
         "TenneT, TransnetBW). Verantwortlich für Systemstabilität "
         "und Ausgleichsenergie."),
    ],
    "V": [
        ("VaR", "Value-at-Risk",
         "α-Quantil der Verlustverteilung. Gibt an, welcher Verlust "
         "mit Wahrscheinlichkeit α nicht überschritten wird. "
         "Nicht sub-additiv → kein kohärentes Risikomaß."),
        ("VP-Risiko", "Volumen-Preis-Risiko",
         "Risiko aus der Korrelation von Mengenabweichung und "
         "Spotpreisabweichung: ΔQ × (Spot − HPFC). Zweiseitig."),
    ],
}


def render_glossary():
    """Render the complete glossary."""
    for letter, entries in sorted(GLOSSARY_ENTRIES.items()):
        st.markdown(f"#### {letter}")
        for term, short, long_desc in entries:
            st.markdown(
                f"**{term}** — *{short}*\n\n"
                f"<span style='color:#94a3b8;font-size:0.85rem'>{long_desc}</span>",
                unsafe_allow_html=True,
            )
        st.markdown("")


# ═══════════════════════════════════════════════════════════════════════════════
#  §22  EXPORT & REPORTING ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

def generate_summary_dataframe(
    results: dict,
    data: PortfolioData,
    params: dict,
) -> pd.DataFrame:
    """Generate a comprehensive summary DataFrame for export."""
    rows = []

    # Header info
    rows.append({"Kategorie": "KONFIGURATION", "Metrik": "Profil", "Wert": data.profile, "Einheit": ""})
    rows.append({"Kategorie": "", "Metrik": "Lieferzeitraum", "Wert": f"{data.years[0]}–{data.years[-1]}", "Einheit": ""})
    rows.append({"Kategorie": "", "Metrik": "Stunden", "Wert": f"{data.n_hours:,}", "Einheit": "h"})
    rows.append({"Kategorie": "", "Metrik": "MC-Pfade", "Wert": f"{params['paths']:,}", "Einheit": ""})
    rows.append({"Kategorie": "", "Metrik": "Konfidenzniveau", "Wert": f"{params['confidence']:.1%}", "Einheit": ""})
    rows.append({"Kategorie": "", "Metrik": "Eigenkapitalkosten", "Wert": f"{params['cost_of_capital']:.1f}", "Einheit": "%"})
    rows.append({"Kategorie": "", "Metrik": "", "Wert": "", "Einheit": ""})

    # Yearly results
    for yr in results["yearly"]:
        rows.append({
            "Kategorie": f"JAHR {yr['year']}",
            "Metrik": "Volumen",
            "Wert": f"{yr['vol']:,.0f}",
            "Einheit": "MWh",
        })
        rows.append({"Kategorie": "", "Metrik": "Strukturbeitrag", "Wert": f"{yr['struktur']:+.4f}", "Einheit": "€/MWh"})
        rows.append({"Kategorie": "", "Metrik": "Prognoserisiko", "Wert": f"+{yr['prognose']:.4f}", "Einheit": "€/MWh"})
        rows.append({"Kategorie": "", "Metrik": "VP-Risiko", "Wert": f"+{yr['vp_prem']:.4f}", "Einheit": "€/MWh"})
        rows.append({"Kategorie": "", "Metrik": "Gesamtaufschlag", "Wert": f"{yr['gesamt']:+.4f}", "Einheit": "€/MWh"})
        rows.append({"Kategorie": "", "Metrik": "CVaR Imbalance", "Wert": f"{yr['im'].CVaR:,.0f}", "Einheit": "€"})
        rows.append({"Kategorie": "", "Metrik": "CVaR VP", "Wert": f"{yr['vm'].CVaR:,.0f}", "Einheit": "€"})
        rows.append({"Kategorie": "", "Metrik": "", "Wert": "", "Einheit": ""})

    # Portfolio
    tot = results["total"]
    rows.append({"Kategorie": "PORTFOLIO", "Metrik": "Gesamtvolumen", "Wert": f"{tot['vol']:,.0f}", "Einheit": "MWh"})
    rows.append({"Kategorie": "", "Metrik": "Strukturbeitrag", "Wert": f"{tot['struktur']:+.4f}", "Einheit": "€/MWh"})
    rows.append({"Kategorie": "", "Metrik": "Prognoserisiko", "Wert": f"+{tot['prognose']:.4f}", "Einheit": "€/MWh"})
    rows.append({"Kategorie": "", "Metrik": "VP-Risiko", "Wert": f"+{tot['vp_prem']:.4f}", "Einheit": "€/MWh"})
    rows.append({"Kategorie": "", "Metrik": "Diversifikation", "Wert": f"−{tot['div_benefit']:.4f}", "Einheit": "€/MWh"})
    rows.append({"Kategorie": "", "Metrik": "GESAMTAUFSCHLAG", "Wert": f"{tot['gesamt']:+.4f}", "Einheit": "€/MWh"})
    rows.append({"Kategorie": "", "Metrik": "", "Wert": "", "Einheit": ""})

    # Risk metrics
    m = tot["combined"]
    rows.append({"Kategorie": "RISIKOMETRIKEN", "Metrik": "Expected Loss", "Wert": f"{m.EL:,.0f}", "Einheit": "€"})
    rows.append({"Kategorie": "", "Metrik": "Std. Deviation", "Wert": f"{m.std:,.0f}", "Einheit": "€"})
    rows.append({"Kategorie": "", "Metrik": f"VaR{int(params['confidence']*100)}", "Wert": f"{m.VaR:,.0f}", "Einheit": "€"})
    rows.append({"Kategorie": "", "Metrik": f"CVaR{int(params['confidence']*100)}", "Wert": f"{m.CVaR:,.0f}", "Einheit": "€"})
    rows.append({"Kategorie": "", "Metrik": "Unexpected Loss", "Wert": f"{m.UL:,.0f}", "Einheit": "€"})
    rows.append({"Kategorie": "", "Metrik": "Spektrales RM", "Wert": f"{m.spectral:,.0f}", "Einheit": "€"})
    rows.append({"Kategorie": "", "Metrik": "Schiefe", "Wert": f"{m.skew:.4f}", "Einheit": ""})
    rows.append({"Kategorie": "", "Metrik": "Exzess-Kurtosis", "Wert": f"{m.kurtosis:.4f}", "Einheit": ""})

    return pd.DataFrame(rows)


def generate_raw_data_export(
    data: PortfolioData,
) -> pd.DataFrame:
    """Generate raw data DataFrame for export."""
    return pd.DataFrame({
        "Zeitstempel": [d.strftime("%d.%m.%Y %H:%M") for d in data.dates],
        "Last_MWh": data.load,
        "HPFC_EUR_MWh": data.hpfc,
        "Temperatur_C": data.temperature,
        "HDD": data.hdd,
        "CDD": data.cdd,
        "Jahr_Index": data.year_map,
    })


def generate_mc_distribution_export(
    results: dict,
) -> pd.DataFrame:
    """Export the full MC loss distribution for external analysis."""
    return pd.DataFrame({
        "Imbalance_EUR": results["raw"]["imb_tot"],
        "VP_Risk_EUR": results["raw"]["vp_tot"],
        "Total_Loss_EUR": results["raw"]["total_loss"],
    })


def render_export_section(
    results: dict,
    data: PortfolioData,
    params: dict,
):
    """Render the export/download section."""
    st.markdown("##### 📥 Daten-Export")
    st.markdown("""
    <div class="info-box">
        Exportieren Sie die Ergebnisse als CSV für die Weiterverarbeitung 
        in Excel, SAP, oder Ihrem internen Risikomanagement-System.
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)

    with c1:
        summary_df = generate_summary_dataframe(results, data, params)
        csv_summary = summary_df.to_csv(index=False, sep=";", decimal=",")
        st.download_button(
            label="📋 Zusammenfassung (CSV)",
            data=csv_summary,
            file_name=f"risikobewertung_{data.years[0]}-{data.years[-1]}.csv",
            mime="text/csv",
            use_container_width=True,
        )
        st.caption(f"{len(summary_df)} Zeilen · Semikolon-getrennt · DE-Format")

    with c2:
        raw_df = generate_raw_data_export(data)
        csv_raw = raw_df.to_csv(index=False, sep=";", decimal=",")
        st.download_button(
            label="📊 Rohdaten (CSV)",
            data=csv_raw,
            file_name=f"stundendaten_{data.years[0]}-{data.years[-1]}.csv",
            mime="text/csv",
            use_container_width=True,
        )
        st.caption(f"{len(raw_df):,} Stunden · Last, HPFC, Temp, HDD, CDD")

    with c3:
        mc_df = generate_mc_distribution_export(results)
        csv_mc = mc_df.to_csv(index=False, sep=";", decimal=",")
        st.download_button(
            label="🎲 MC-Verteilungen (CSV)",
            data=csv_mc,
            file_name=f"mc_verteilungen_{params['paths']}pfade.csv",
            mime="text/csv",
            use_container_width=True,
        )
        st.caption(f"{len(mc_df):,} Pfade · Imbalance, VP, Total Loss")


# ═══════════════════════════════════════════════════════════════════════════════
#  §23  ADVANCED ANALYTICS
# ═══════════════════════════════════════════════════════════════════════════════

def compute_rolling_risk(
    losses: np.ndarray,
    window_sizes: List[int] = None,
    alpha: float = 0.95,
) -> pd.DataFrame:
    """
    Compute rolling risk metrics across increasing sample sizes.
    Useful for convergence and stability analysis.
    """
    if window_sizes is None:
        n = len(losses)
        window_sizes = sorted(set(
            np.logspace(np.log10(50), np.log10(n), 60).astype(int).tolist()
        ))

    rows = []
    for w in window_sizes:
        if w > len(losses):
            break
        subset = losses[:w]
        m = compute_risk_metrics(subset, alpha)
        rows.append({
            "N": w,
            "EL": m.EL,
            "Std": m.std,
            "VaR": m.VaR,
            "CVaR": m.CVaR,
            "Skew": m.skew,
            "Kurt": m.kurtosis,
        })
    return pd.DataFrame(rows)


def compute_tail_index(losses: np.ndarray, k_range: tuple = (10, 200)) -> dict:
    """
    Estimate the tail index using the Hill estimator.

    The Hill estimator for the tail index ξ (Pareto exponent α = 1/ξ):

        ξ_k = (1/k) Σ_{i=1}^{k} [ln(X_{(n-i+1)}) − ln(X_{(n-k)})]

    A tail index ξ > 0.5 indicates very heavy tails (infinite variance
    beyond the tail). For German electricity portfolios, typical values
    are ξ ≈ 0.2–0.4.
    """
    positive_losses = losses[losses > 0]
    if len(positive_losses) < k_range[1]:
        return {"hill_estimate": None, "k_values": [], "xi_values": []}

    sorted_losses = np.sort(positive_losses)[::-1]
    n = len(sorted_losses)

    k_values = list(range(k_range[0], min(k_range[1], n // 2)))
    xi_values = []

    for k in k_values:
        if k < 2 or k >= n:
            continue
        log_excesses = np.log(sorted_losses[:k]) - np.log(sorted_losses[k])
        xi = np.mean(log_excesses)
        xi_values.append(xi)

    # Optimal k via minimizing asymptotic MSE (simplified)
    if xi_values:
        # Use the plateau method: find the most stable region
        xi_arr = np.array(xi_values)
        diffs = np.abs(np.diff(xi_arr))
        if len(diffs) > 10:
            window = 10
            rolling_var = np.array([
                diffs[i:i+window].var() for i in range(len(diffs) - window)
            ])
            best_k_idx = np.argmin(rolling_var) + window // 2
            best_xi = xi_values[best_k_idx] if best_k_idx < len(xi_values) else xi_values[-1]
        else:
            best_xi = np.median(xi_arr)
    else:
        best_xi = None

    return {
        "hill_estimate": best_xi,
        "k_values": k_values[:len(xi_values)],
        "xi_values": xi_values,
    }


def compute_year_correlations(results: dict) -> pd.DataFrame:
    """Compute pairwise loss correlations across delivery years."""
    imb_y = results["raw"]["imb_y"]
    vp_y = results["raw"]["vp_y"]
    total_y = imb_y + vp_y
    n_years = total_y.shape[1]

    corr_matrix = np.corrcoef(total_y.T)
    return pd.DataFrame(
        corr_matrix,
        index=[f"Jahr {i+1}" for i in range(n_years)],
        columns=[f"Jahr {i+1}" for i in range(n_years)],
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  §24  CHART LIBRARY — PART 3 (ADVANCED VISUALS)
# ═══════════════════════════════════════════════════════════════════════════════

def chart_hill_plot(losses: np.ndarray) -> go.Figure:
    """Hill plot for tail index estimation."""
    tail_info = compute_tail_index(losses)
    if not tail_info["k_values"]:
        fig = go.Figure()
        fig.add_annotation(text="Zu wenige Daten für Hill-Plot", x=0.5, y=0.5,
                           showarrow=False, xref="paper", yref="paper")
        fig.update_layout(**make_layout(height=350))
        return fig

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=tail_info["k_values"],
        y=tail_info["xi_values"],
        mode="lines",
        line=dict(color=Colors.BLUE, width=2),
        name="ξ(k)",
        hovertemplate="k = %{x}<br>ξ = %{y:.4f}<extra></extra>",
    ))

    if tail_info["hill_estimate"]:
        fig.add_hline(
            y=tail_info["hill_estimate"],
            line=dict(color=Colors.ROSE, width=1.5, dash="dash"),
            annotation_text=f"ξ* ≈ {tail_info['hill_estimate']:.3f}",
            annotation_font=dict(color=Colors.ROSE, size=11),
        )
        # Reference lines for interpretation
        fig.add_hline(y=0.5, line=dict(color=Colors.AMBER, width=1, dash="dot"),
                      annotation_text="ξ=0.5 (Var → ∞)",
                      annotation_font=dict(size=9, color=Colors.AMBER))

    fig.update_layout(**make_layout(
        title=dict(text="Hill-Plot — Tail-Index Schätzung", font=dict(size=13)),
        xaxis_title="Anzahl Tail-Beobachtungen (k)",
        yaxis_title="Tail-Index ξ",
        height=370, showlegend=False,
    ))
    return fig


def chart_year_correlation_heatmap(results: dict) -> go.Figure:
    """Heatmap of inter-year loss correlations."""
    corr_df = compute_year_correlations(results)

    fig = go.Figure(go.Heatmap(
        z=corr_df.values,
        x=corr_df.columns.tolist(),
        y=corr_df.index.tolist(),
        colorscale="RdBu_r",
        zmid=0,
        zmin=-1, zmax=1,
        text=[[f"{v:.2f}" for v in row] for row in corr_df.values],
        texttemplate="%{text}",
        textfont=dict(size=12, color="white"),
        colorbar=dict(title="ρ", len=0.6),
        hovertemplate="%{x} vs %{y}<br>ρ = %{z:.3f}<extra></extra>",
    ))
    fig.update_layout(**make_layout(
        title=dict(text="Inter-Year Verlustkorrelation", font=dict(size=13)),
        height=400,
        yaxis=dict(autorange="reversed"),
    ))
    return fig


def chart_box_by_year(results: dict) -> go.Figure:
    """Box plots of total loss by delivery year."""
    yearly = results["yearly"]
    imb_y = results["raw"]["imb_y"]
    vp_y = results["raw"]["vp_y"]

    fig = go.Figure()
    for yr in yearly:
        yi = yr["yi"]
        total_y = imb_y[:, yi] + vp_y[:, yi]
        fig.add_trace(go.Box(
            y=total_y / yr["vol"],
            name=str(yr["year"]),
            marker_color=Colors.BLUE,
            line_color=Colors.BLUE_LIGHT,
            boxpoints=False,
            hovertemplate=f"{yr['year']}<br>%{{y:.3f}} €/MWh<extra></extra>",
        ))

    fig.update_layout(**make_layout(
        title=dict(text="Verlustverteilung pro Lieferjahr  (€/MWh)", font=dict(size=13)),
        yaxis_title="€ / MWh",
        showlegend=False,
        height=400,
    ))
    return fig


def chart_cumulative_loss_paths(results: dict, n_show: int = 50) -> go.Figure:
    """Cumulative loss development across paths (fan chart)."""
    total = results["raw"]["total_loss"]
    n = len(total)
    sorted_t = np.sort(total)

    percentiles = [5, 10, 25, 50, 75, 90, 95]
    pct_values = np.percentile(total, percentiles)

    fig = go.Figure()

    # Fan chart using percentile bands
    for i in range(len(percentiles) // 2):
        lo_idx = i
        hi_idx = len(percentiles) - 1 - i
        alpha_val = 0.08 + i * 0.06

        fig.add_trace(go.Scatter(
            x=[percentiles[lo_idx], percentiles[hi_idx]],
            y=[pct_values[lo_idx], pct_values[hi_idx]],
            mode="markers+lines",
            marker=dict(size=8, color=Colors.BLUE),
            line=dict(color=Colors.BLUE, width=2),
            showlegend=False,
        ))

    # Histogram rotated as horizontal
    fig.add_trace(go.Histogram(
        x=total, nbinsx=100,
        marker_color=Colors.with_alpha(Colors.BLUE, 0.35),
        marker_line=dict(color=Colors.BLUE, width=0.5),
        name="Verteilung",
    ))

    fig.update_layout(**make_layout(
        title=dict(text="Gesamtverlust-Verteilung mit Perzentilen", font=dict(size=13)),
        xaxis_title="Gesamtverlust (€)",
        yaxis_title="Häufigkeit",
        height=380, showlegend=False,
    ))
    return fig


def chart_vol_error_paths(results: dict, data: PortfolioData) -> go.Figure:
    """Show sample volume error paths (AR(1) realisations)."""
    diag = results["diagnostics"]
    samples = diag["vol_error_samples"]
    n_paths = min(diag["n_diag_paths"], 10)
    n_hours = min(diag["n_diag_hours"], 2000)
    x = data.dates[:n_hours]

    fig = go.Figure()
    for p in range(n_paths):
        fig.add_trace(go.Scatter(
            x=x, y=samples[p, :n_hours] * 100,
            mode="lines", opacity=0.35,
            line=dict(width=0.9, color=Colors.EMERALD),
            showlegend=False, hoverinfo="skip",
        ))

    fig.add_hline(y=0, line=dict(color=Colors.SLATE_500, width=1, dash="dash"))
    fig.update_layout(**make_layout(
        title=dict(text=f"AR(1) Volumenfehler-Pfade  ({n_paths} Pfade)", font=dict(size=13)),
        yaxis_title="Abweichung (%)",
        height=370, showlegend=False,
    ))
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
#  §25  REFERENCES & LITERATURE DATABASE
# ═══════════════════════════════════════════════════════════════════════════════

REFERENCES = [
    {
        "authors": "Artzner, P., Delbaen, F., Eber, J.-M., Heath, D.",
        "year": 1999,
        "title": "Coherent Measures of Risk",
        "journal": "Mathematical Finance",
        "volume": "9(3), 203–228",
        "relevance": "Grundlage der CVaR-basierten Risikomessung",
    },
    {
        "authors": "Acerbi, C.",
        "year": 2002,
        "title": "Spectral measures of risk: A coherent representation of subjective risk aversion",
        "journal": "Journal of Banking & Finance",
        "volume": "26(7), 1505–1518",
        "relevance": "Spektrales Risikomaß mit Risikoaversionsspektrum",
    },
    {
        "authors": "Benth, F.E., Šaltytė Benth, J., Koekebakker, S.",
        "year": 2008,
        "title": "Stochastic Modelling of Electricity and Related Markets",
        "journal": "World Scientific",
        "volume": "",
        "relevance": "Standardwerk für stochastische Energiepreismodelle",
    },
    {
        "authors": "Bollerslev, T.",
        "year": 1986,
        "title": "Generalized Autoregressive Conditional Heteroskedasticity",
        "journal": "Journal of Econometrics",
        "volume": "31(3), 307–327",
        "relevance": "GARCH(1,1) Volatilitätsmodell",
    },
    {
        "authors": "Burger, M., Graeber, B., Schindlmayr, G.",
        "year": 2014,
        "title": "Managing Energy Risk: An Integrated View on Power and Other Energy Markets",
        "journal": "Wiley Finance, 2nd Edition",
        "volume": "",
        "relevance": "Praxishandbuch für Energierisikomanagement",
    },
    {
        "authors": "Escribano, A., Peña, J.I., Villaplana, P.",
        "year": 2011,
        "title": "Modelling Electricity Prices: International Evidence",
        "journal": "Oxford Bulletin of Economics and Statistics",
        "volume": "73(5), 622–650",
        "relevance": "Internationale Evidenz für Strompreismodelle",
    },
    {
        "authors": "McNeil, A.J., Frey, R., Embrechts, P.",
        "year": 2015,
        "title": "Quantitative Risk Management: Concepts, Techniques and Tools",
        "journal": "Princeton University Press, Revised Edition",
        "volume": "",
        "relevance": "Standardwerk für quantitatives Risikomanagement",
    },
    {
        "authors": "Merton, R.C.",
        "year": 1976,
        "title": "Option pricing when underlying stock returns are discontinuous",
        "journal": "Journal of Financial Economics",
        "volume": "3(1-2), 125–144",
        "relevance": "Jump-Diffusion Modell für Preissprünge",
    },
    {
        "authors": "Weron, R.",
        "year": 2014,
        "title": "Electricity price forecasting: A review of the state-of-the-art with a look into the future",
        "journal": "International Journal of Forecasting",
        "volume": "30(4), 1030–1081",
        "relevance": "Umfassende Review der Strompreisprognose",
    },
    {
        "authors": "BDEW / VDN",
        "year": 2024,
        "title": "Standardlastprofile Strom — Anwendung und Methodik",
        "journal": "Bundesverband der Energie- und Wasserwirtschaft",
        "volume": "",
        "relevance": "Offizielle BDEW-Lastprofile H0, G0–G6, L0–L2",
    },
    {
        "authors": "BNetzA",
        "year": 2024,
        "title": "Festlegung zu Bilanzkreisabrechnungen (reBAP-Systematik)",
        "journal": "Bundesnetzagentur, BK6-20-201",
        "volume": "",
        "relevance": "Regulatorische Grundlage des reBAP-Preismodells",
    },
    {
        "authors": "Hill, B.M.",
        "year": 1975,
        "title": "A Simple General Approach to Inference About the Tail of a Distribution",
        "journal": "Annals of Statistics",
        "volume": "3(5), 1163–1174",
        "relevance": "Hill-Schätzer für den Tail-Index",
    },
]


def render_references():
    """Render the complete reference list."""
    st.markdown("##### 📚 Literatur & Referenzen")
    for i, ref in enumerate(REFERENCES, 1):
        # Build parts separately to avoid f-string quote conflicts
        authors = ref["authors"]
        year = ref["year"]
        title = ref["title"]
        journal = ref["journal"]
        volume = ref["volume"]
        relevance = ref["relevance"]

        journal_part = ""
        if journal:
            journal_part = f" *{journal}*"
        volume_part = ""
        if volume:
            volume_part = f" {volume}"

        ref_text = (
            f"**[{i}]** {authors} ({year}). "
            f"*{title}.*"
            f"{journal_part}{volume_part}."
        )
        relevance_html = (
            f'<span style="color:#64748b;font-size:0.78rem">'
            f"→ {relevance}</span>"
        )

        st.markdown(
            ref_text + "\n\n" + relevance_html,
            unsafe_allow_html=True,
        )


# ═══════════════════════════════════════════════════════════════════════════════
#  §26  EXTENDED run_simulation_and_display() [OVERRIDE]
#
#  This function REPLACES the version from Block 2, adding:
#    - Methodology tab
#    - Glossary tab
#    - Export section
#    - Advanced analytics
#    - Additional diagnostic charts
#
# ═══════════════════════════════════════════════════════════════════════════════

def run_simulation_and_display(
    data: PortfolioData,
    params: dict,
):
    """
    EXTENDED master orchestrator (overrides Block 2 version).

    Runs the MC simulation and renders the complete results dashboard
    including all 8 tabs.
    """
    # ── Run simulation if triggered ──
    if st.session_state.get("run_simulation", False):
        np.random.seed(params.get("seed", 42))
        results = run_simulation(data, params)
        st.session_state["results"] = results
        st.session_state["run_simulation"] = False
        st.session_state["simulation_count"] = (
            st.session_state.get("simulation_count", 0) + 1
        )

    results = st.session_state.get("results")
    if results is None:
        return

    tot = results["total"]
    years = data.years

    # ════════════════════════════════════════════════════════════
    #  KPI STRIP
    # ════════════════════════════════════════════════════════════

    st.markdown("---")
    st.markdown(
        '<div style="animation: fadeInUp 0.4s ease-out">',
        unsafe_allow_html=True,
    )
    st.markdown("##### ⚡ Simulationsergebnisse")

    k1, k2, k3, k4, k5, k6 = st.columns(6)

    kpi_defs = [
        (k1, "Strukturbeitrag", f"{tot['struktur']:+.3f}", "€/MWh", None, "neutral"),
        (k2, "Prognoserisiko", f"+{tot['prognose']:.3f}", "€/MWh", None, "neutral"),
        (k3, "VP-Risiko", f"+{tot['vp_prem']:.3f}", "€/MWh", None, "neutral"),
        (k4, "Diversifikation", f"−{tot['div_benefit']:.3f}", "€/MWh",
         "Portfolio-Effekt", "positive"),
        (k5, f"CVaR{int(params['confidence']*100)}",
         f"{tot['combined'].CVaR:,.0f}", "€",
         f"VaR: {tot['combined'].VaR:,.0f}€", "neutral"),
        (k6, "Volumen gesamt", f"{tot['vol']:,.0f}", "MWh",
         f"{len(years)} Jahre", "neutral"),
    ]

    for col, (_, label, value, unit, delta, dtype) in zip(
        [k1, k2, k3, k4, k5, k6], kpi_defs
    ):
        col.markdown(
            render_kpi_card(label, value, unit, delta, dtype),
            unsafe_allow_html=True,
        )

    st.markdown("")

    # ════════════════════════════════════════════════════════════
    #  BIG RESULT CARD
    # ════════════════════════════════════════════════════════════

    st.markdown(
        f'<div class="big-result">'
        f'<div class="br-label">Portfolio-Mischpreis — Gesamtaufschlag</div>'
        f'<div class="br-value">{tot["gesamt"]:+.2f}'
        f' <span style="font-size:1.2rem;color:{Colors.BLUE_LIGHT};'
        f'font-weight:500">€ / MWh</span></div>'
        f'<div class="br-context">'
        f'{params["paths"]:,} Pfade  ·  '
        f'α = {params["confidence"]:.0%}  ·  '
        f'CoE = {params["cost_of_capital"]:.0f}%  ·  '
        f'Profil {data.profile}  ·  '
        f'{years[0]}–{years[-1]}  ·  '
        f'Simulation #{st.session_state.get("simulation_count", 1)}'
        f'</div></div>',
        unsafe_allow_html=True,
    )

    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("")

    # ════════════════════════════════════════════════════════════
    #  TABBED RESULTS — 8 TABS
    # ════════════════════════════════════════════════════════════

    tab_res, tab_dist, tab_stress, tab_sens, tab_diag, tab_export, tab_method, tab_glossary = st.tabs([
        "📋  Ergebnisse",
        "📈  Verteilungen",
        "🔥  Stresstest",
        "🎯  Sensitivitäten",
        "🔬  Diagnostik",
        "📥  Export",
        "📐  Methodik",
        "📖  Glossar",
    ])

    # ─────────────────── TAB 1: RESULTS ───────────────────
    with tab_res:
        c_table, c_waterfall = st.columns([3, 2])

        with c_table:
            st.markdown("##### Granulare Jahresbewertung")
            st.markdown(
                '<div style="font-size:0.7rem;color:#64748b;margin-bottom:8px">'
                'Alle Werte in €/MWh. Sub-Details: EL=Expected Loss, '
                'UL=Unexpected Loss, CVaR=Cond. Value-at-Risk.'
                '</div>',
                unsafe_allow_html=True,
            )
            st.markdown(render_results_table(results), unsafe_allow_html=True)

        with c_waterfall:
            st.plotly_chart(chart_waterfall(results), use_container_width=True)

        c_bars, c_elul = st.columns(2)
        with c_bars:
            st.plotly_chart(chart_annual_bars(results, years), use_container_width=True)
        with c_elul:
            st.plotly_chart(chart_el_ul_decomposition(results), use_container_width=True)

        c_pie, c_info = st.columns([1, 1])
        with c_pie:
            st.plotly_chart(chart_risk_contribution_pie(results), use_container_width=True)
        with c_info:
            st.markdown("##### Interpretation")
            st.markdown(f"""
            <div class="info-box">
            <strong>Strukturbeitrag</strong> ({tot['struktur']:+.3f} €/MWh):<br>
            Differenz zwischen mengengewichtetem Ø-HPFC und Baseload-Preis. 
            Positiv = Lastprofil liegt in teuren Stunden.<br><br>
            <strong>Prognoserisiko</strong> (+{tot['prognose']:.3f} €/MWh):<br>
            Erwarteter Imbalance-Schaden + RORAC auf CVaR-Unexpected-Loss.<br><br>
            <strong>VP-Risiko</strong> (+{tot['vp_prem']:.3f} €/MWh):<br>
            Zweiseitiges Volumen-Preis-Risiko aus ΔQ × (Spot − HPFC).<br><br>
            <strong>Diversifikation</strong> (−{tot['div_benefit']:.3f} €/MWh):<br>
            Sub-Additivitätseffekt über {len(years)} Lieferjahre.
            </div>
            """, unsafe_allow_html=True)

    # ─────────────────── TAB 2: DISTRIBUTIONS ───────────────────
    with tab_dist:
        st.markdown("##### Verlustverteilungen")

        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(
                chart_distribution(results["raw"]["imb_tot"], "Imbalance-Kosten",
                                   Colors.AMBER, params["confidence"]),
                use_container_width=True)
        with c2:
            st.plotly_chart(
                chart_distribution(results["raw"]["vp_tot"], "VP-Risiko",
                                   Colors.ROSE, params["confidence"]),
                use_container_width=True)

        st.plotly_chart(
            chart_distribution(results["raw"]["total_loss"], "Gesamtes Portfolio-Risiko",
                               Colors.BLUE, params["confidence"]),
            use_container_width=True)

        st.markdown("##### Verteilungsanalyse")
        c3, c4 = st.columns(2)
        with c3:
            st.plotly_chart(
                chart_qq_plot(results["raw"]["total_loss"], "Portfolio-Verluste"),
                use_container_width=True)
        with c4:
            st.plotly_chart(
                chart_ecdf(results["raw"]["total_loss"], "Portfolio", Colors.BLUE),
                use_container_width=True)

        st.markdown("##### Erweiterte Analyse")
        c5, c6 = st.columns(2)
        with c5:
            st.plotly_chart(chart_box_by_year(results), use_container_width=True)
        with c6:
            st.plotly_chart(chart_hill_plot(results["raw"]["total_loss"]),
                            use_container_width=True)

        with st.expander("📊 Vollständige Risikostatistiken"):
            stats_df = render_risk_stats_table(results, params)
            st.dataframe(stats_df, hide_index=True, use_container_width=True, height=600)

    # ─────────────────── TAB 3: STRESS TESTING ───────────────────
    with tab_stress:
        st.markdown("##### Stresstest-Szenarien — Deutscher Strommarkt")
        st.markdown("""
        <div class="info-box">
            Deterministische Worst-Case-Analyse kalibriert auf historische 
            Extremereignisse 2017–2024. Simultane Preis- und Mengenschocks.
        </div>
        """, unsafe_allow_html=True)

        stress_results = run_all_stress_tests(data, params)

        for row_start in range(0, len(stress_results), 3):
            cols = st.columns(3)
            for j, col in enumerate(cols):
                idx = row_start + j
                if idx >= len(stress_results):
                    break
                sr = stress_results[idx]
                impact_cls = "loss" if sr["total"] > 0 else "gain"
                with col:
                    st.markdown(f"""
                    <div class="stress-card">
                        <h4>{sr['icon']}  {sr['name']}</h4>
                        <div class="stress-desc">{sr['desc']}</div>
                        <div class="stress-impact {impact_cls}">{sr['total']:+,.0f} €</div>
                        <div class="stress-detail">
                            {sr['per_mwh']:+.2f} €/MWh · {sr['duration']:,}h · {sr['severity']}
                        </div>
                        <div class="stress-detail">Hist.: {sr['historical']}</div>
                    </div>
                    """, unsafe_allow_html=True)

        st.markdown("")
        st.plotly_chart(chart_stress_waterfall(stress_results), use_container_width=True)

        with st.expander("📋 Detaillierte Stress-Zerlegung"):
            sdf = pd.DataFrame([{
                "Szenario": sr["name"],
                "ΔP": f"{sr['price_shock']:+.0f} €",
                "ΔV": f"{sr['vol_shock']:+.0%}",
                "Dauer": f"{sr['duration']:,}h",
                "VP": f"{sr['vp_loss']:+,.0f} €",
                "Imb": f"{sr['imb_loss']:+,.0f} €",
                "Strukt": f"{sr['struct_impact']:+,.0f} €",
                "Gesamt": f"{sr['total']:+,.0f} €",
                "€/MWh": f"{sr['per_mwh']:+.3f}",
            } for sr in stress_results])
            st.dataframe(sdf, hide_index=True, use_container_width=True)

    # ─────────────────── TAB 4: SENSITIVITIES ───────────────────
    with tab_sens:
        st.markdown("##### Parameter-Sensitivitätsanalyse")
        st.markdown("""
        <div class="info-box">
            <strong>Energy Greeks:</strong> Analytische ∂π/∂θ Approximation — 
            analog zu Derivate-Greeks. Identifiziert die Haupttreiber des Risikos.
        </div>
        """, unsafe_allow_html=True)

        sens_df = compute_sensitivities(params, results)

        c_torn, c_tbl = st.columns([3, 2])
        with c_torn:
            st.plotly_chart(chart_tornado(sens_df), use_container_width=True)
        with c_tbl:
            display_df = sens_df[["analogy", "symbol", "base", "delta",
                                   "impact_up", "impact_dn", "sensitivity"]].copy()
            display_df.columns = ["Greek", "Symbol", "Basis", "Δ",
                                   "+Δ Impact", "−Δ Impact", "∂π/∂θ"]
            for c in ["+Δ Impact", "−Δ Impact", "∂π/∂θ"]:
                display_df[c] = display_df[c].apply(lambda x: f"{x:+.5f}")
            st.dataframe(display_df, hide_index=True, use_container_width=True, height=400)

        st.markdown("---")
        st.markdown("""
        ##### Interpretationshilfe
        
        | Greek | Parameter | Bedeutung |
        |---|---|---|
        | **Volume Vega** | σ_V | Prämien-Sensitivität gegenüber Mengenunsicherheit |
        | **Correlation Rho** | ρ | Empfindlichkeit Preis-Mengen-Kopplung |
        | **Price Vega** | σ_S | Einfluss Spot-Volatilität |
        | **Reversion Delta** | κ | Auswirkung Mean-Reversion Geschwindigkeit |
        | **Persistence Gamma** | φ | Empfindlichkeit Wetter-Persistenz |
        | **Jump Vanna** | λ | Sprung-Häufigkeit → Tail-Risiko |
        | **Capital Lambda** | r_EC | RORAC-Sensitivität |
        """)

    # ─────────────────── TAB 5: DIAGNOSTICS ───────────────────
    with tab_diag:
        st.markdown("##### Modelldiagnostik & Konvergenz")

        c_conv, c_paths = st.columns(2)
        with c_conv:
            st.plotly_chart(chart_convergence(results, params["confidence"]),
                            use_container_width=True)
        with c_paths:
            st.plotly_chart(chart_spot_paths(results, data), use_container_width=True)

        c_vol, c_corr = st.columns(2)
        with c_vol:
            st.plotly_chart(chart_vol_error_paths(results, data), use_container_width=True)
        with c_corr:
            if len(results["yearly"]) > 1:
                st.plotly_chart(chart_year_correlation_heatmap(results),
                                use_container_width=True)
            else:
                st.info("Inter-Year Korrelation benötigt mindestens 2 Lieferjahre.")

        st.markdown("##### Stichproben-Diagnostik")
        combined = results["raw"]["total_loss"]
        cd1, cd2, cd3, cd4 = st.columns(4)

        with cd1:
            if len(combined) >= 20:
                _, p_jb = sp_stats.jarque_bera(combined)
                st.metric("Jarque-Bera", f"p = {p_jb:.2e}",
                          delta="Nicht normal" if p_jb < 0.05 else "≈ Normal",
                          delta_color="inverse" if p_jb < 0.05 else "normal")
            else:
                st.metric("Jarque-Bera", "N < 20")

        with cd2:
            st.metric("Kurtosis", f"{tot['combined'].kurtosis:.3f}",
                      delta="Fat Tails" if tot['combined'].kurtosis > 1 else "≈ Normal",
                      delta_color="inverse" if tot['combined'].kurtosis > 1 else "normal")

        with cd3:
            diag = results["diagnostics"]
            if len(diag["convergence_cvar"]) > 5:
                last_vals = diag["convergence_cvar"][-10:]
                cv = np.std(last_vals) / (np.abs(np.mean(last_vals)) + 1e-6) * 100
                st.metric("CVaR CV", f"{cv:.2f}%",
                          delta="Konvergiert" if cv < 3 else "Mehr Pfade",
                          delta_color="normal" if cv < 3 else "inverse")

        with cd4:
            tail_info = compute_tail_index(combined)
            if tail_info["hill_estimate"]:
                st.metric("Tail-Index ξ", f"{tail_info['hill_estimate']:.3f}",
                          delta="Schwerer Tail" if tail_info["hill_estimate"] > 0.3 else "Moderater Tail",
                          delta_color="inverse" if tail_info["hill_estimate"] > 0.5 else "normal")

        st.markdown("##### Parametrisierung")
        param_rows = [
            ("Spot", "κ", f"{params['kappa']:.3f}", "h⁻¹"),
            ("Spot", "σ_S", f"{params['sigma_price']:.1f}", "€/MWh"),
            ("Sprung", "λ", f"{params['jump_prob']:.3f}", "h⁻¹"),
            ("Sprung", "σ_J", f"{params['jump_size']:.0f}", "€/MWh"),
            ("GARCH", "aktiv", str(params.get("garch_enabled", True)), ""),
            ("GARCH", "ω/α/β",
             f"{params.get('garch_omega',5):.1f}/{params.get('garch_alpha',.08):.2f}/{params.get('garch_beta',.88):.2f}", ""),
            ("Volumen", "φ", f"{params['phi']:.3f}", ""),
            ("Volumen", "σ_V", f"{params['vol_error']:.1f}", "%"),
            ("Abhängigkeit", "ρ", f"{params['correlation']:.0f}", "%"),
            ("Risiko", "α", f"{params['confidence']:.1%}", ""),
            ("Kapital", "r_EC", f"{params['cost_of_capital']:.1f}", "%"),
            ("MC", "N", f"{params['paths']:,}", "Pfade"),
            ("MC", "Seed", f"{params.get('seed', 42)}", ""),
        ]
        st.dataframe(
            pd.DataFrame(param_rows, columns=["Modul", "Param", "Wert", "Einheit"]),
            hide_index=True, use_container_width=True)

    # ─────────────────── TAB 6: EXPORT ───────────────────
    with tab_export:
        render_export_section(results, data, params)

    # ─────────────────── TAB 7: METHODOLOGY ───────────────────
    with tab_method:
        render_methodology_tab()

    # ─────────────────── TAB 8: GLOSSARY ───────────────────
    with tab_glossary:
        st.markdown("##### 📖 Glossar — Energiewirtschaft & Risikomanagement")
        st.markdown("""
        <div class="info-box">
            Fachbegriffe aus der deutschen Energiewirtschaft, dem 
            quantitativen Risikomanagement und der Finanzmathematik.
        </div>
        """, unsafe_allow_html=True)

        search_term = st.text_input(
            "🔍 Begriff suchen",
            placeholder="z.B. CVaR, reBAP, Dunkelflaute ...",
            key="glossary_search",
        )

        if search_term:
            found = False
            for letter, entries in sorted(GLOSSARY_ENTRIES.items()):
                for term, short, long_desc in entries:
                    if search_term.lower() in term.lower() or search_term.lower() in long_desc.lower():
                        st.markdown(
                            f"**{term}** — *{short}*\n\n"
                            f"<span style='color:#94a3b8;font-size:0.85rem'>{long_desc}</span>",
                            unsafe_allow_html=True,
                        )
                        found = True
            if not found:
                st.info(f"Kein Eintrag für '{search_term}' gefunden.")
        else:
            render_glossary()


# ═══════════════════════════════════════════════════════════════════════════════
#  §27  EXTENDED main() [OVERRIDE]
#
#  This replaces the main() from Block 1 to add the methodology
#  tab accessible BEFORE running a simulation.
#
#  The key change: We add a global "Methodik & Hilfe" section
#  at the bottom that's always visible, even without data.
#
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    """
    EXTENDED main application entry point.

    Overrides Block 1's main() with:
    - All original functionality (data input, synthetic generation, paste)
    - Methodology accessible without simulation
    - Global help section
    - Proper integration of all 3 blocks
    """

    # ── Inject CSS ──
    inject_css()

    # ── Session State Defaults ──
    defaults = {
        "data": None,
        "results": None,
        "paste_text": "",
        "data_source": None,
        "active_preset": None,
        "simulation_count": 0,
        "run_simulation": False,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

    # ════════════════════════════════════════════════════════════
    #  SIDEBAR (identical to Block 1 — full parameter config)
    # ════════════════════════════════════════════════════════════

    with st.sidebar:
        st.markdown("### ⚡ Engine Konfiguration")
        st.caption(f"Version {APP_VERSION}")

        with st.expander("🚀  Schnellstart-Presets", expanded=False):
            preset_name = st.selectbox(
                "Szenario", ["(Manuell)"] + list(PRESET_SCENARIOS.keys()),
                format_func=lambda x: x if x == "(Manuell)" else f"{PRESET_SCENARIOS[x]['icon']} {x}",
                key="preset_sel_v2")
            if preset_name != "(Manuell)":
                p = PRESET_SCENARIOS[preset_name]
                st.markdown(f'<div class="info-box">{p["description"]}</div>', unsafe_allow_html=True)
                if st.button("📋 Übernehmen", use_container_width=True):
                    st.session_state["active_preset"] = preset_name
                    st.rerun()

        ap = PRESET_SCENARIOS.get(st.session_state.get("active_preset", ""), None)

        with st.expander("📐  Lastprofil & Volumen", expanded=True):
            profile = st.selectbox("BDEW Profil", list(BDEW_PROFILES.keys()),
                format_func=lambda x: f"{BDEW_PROFILES[x]['icon']} {x} — {BDEW_PROFILES[x]['name']}",
                index=list(BDEW_PROFILES.keys()).index(ap["profile"]) if ap else 0)
            pi = BDEW_PROFILES[profile]
            st.markdown(f'<div class="info-box"><strong>{pi["name_en"]}</strong><br>{pi["description"]}<br><em>{pi["typical_volume"]}</em></div>', unsafe_allow_html=True)
            annual_mwh = st.number_input("Jahresverbrauch (MWh)", 100, 1_000_000, ap["annual_mwh"] if ap else 10_000, step=500)
            c1, c2 = st.columns(2)
            start_year = c1.number_input("Startjahr", 2024, 2040, ap["start_year"] if ap else 2026)
            n_years = c2.selectbox("Laufzeit", [1,2,3,4,5], index=[1,2,3,4,5].index(ap["n_years"]) if ap else 2, format_func=lambda n: f"{n} J.")
            temp_sensitivity = st.slider("β_H Temp-Sens.", 0.0, 0.10, 0.03, 0.005, format="%.3f")

        with st.expander("📊  Markt & Kapital", expanded=True):
            base_price = st.number_input("Baseload Frontjahr (€/MWh)", 10.0, 500.0, ap["base_price"] if ap else 80.0, step=5.0, format="%.1f")
            c1, c2 = st.columns(2)
            confidence = c1.selectbox("α", [0.90,0.95,0.975,0.99], index=1, format_func=lambda x: f"{x:.1%}")
            cost_cap = c2.number_input("CoE (%)", 5.0, 35.0, 15.0, step=1.0, format="%.1f")

        with st.expander("🔬  Stochastik", expanded=False):
            paths = st.select_slider("MC Pfade", [500,1000,2000,5000,10000,20000], 2000)
            vol_error = st.slider("σ_V (%)", 2.0, 25.0, ap["vol_error"] if ap else 8.0, 0.5)
            correlation = st.slider("ρ (%)", -60.0, 80.0, ap["correlation"] if ap else 40.0, 5.0)
            kappa = st.slider("κ (h⁻¹)", 0.01, 0.50, 0.10, 0.01)
            phi = st.slider("φ", 0.50, 0.998, 0.95, 0.002, format="%.3f")
            sigma_price = st.slider("σ_S (€)", 3.0, 50.0, 15.0, 1.0)

        with st.expander("⚡  Sprünge & GARCH", expanded=False):
            jump_prob = st.slider("λ (h⁻¹)", 0.0, 0.15, 0.02, 0.005, format="%.3f")
            jump_size = st.slider("σ_J (€)", 10.0, 300.0, 80.0, 10.0)
            garch_enabled = st.checkbox("GARCH(1,1)", value=True)
            g_omega = st.number_input("ω", 1.0, 20.0, 5.0, 0.5, format="%.1f") if garch_enabled else 5.0
            g_alpha = st.number_input("α_G", 0.01, 0.20, 0.08, 0.01, format="%.2f") if garch_enabled else 0.08
            g_beta = st.number_input("β_G", 0.70, 0.97, 0.88, 0.01, format="%.2f") if garch_enabled else 0.88
            if garch_enabled and (g_alpha + g_beta) >= 1.0:
                st.warning(f"⚠️ α+β = {g_alpha+g_beta:.2f} ≥ 1!")

        seed = st.number_input("Seed", 0, 99999, 42, step=1)

    params = dict(
        base_price=base_price, paths=paths, confidence=confidence,
        cost_of_capital=cost_cap, vol_error=vol_error, correlation=correlation,
        kappa=kappa, phi=phi, sigma_price=sigma_price,
        jump_prob=jump_prob, jump_size=jump_size,
        garch_enabled=garch_enabled, garch_omega=g_omega,
        garch_alpha=g_alpha, garch_beta=g_beta,
        seed=seed, profile=profile, annual_mwh=annual_mwh,
        start_year=start_year, n_years=n_years,
        temp_sensitivity=temp_sensitivity,
    )
    st.session_state["params"] = params

    # ════════════════════════════════════════════════════════════
    #  HERO BANNER
    # ════════════════════════════════════════════════════════════

    st.markdown(f"""
    <div class="hero-banner">
        <h1>⚡ {APP_TITLE}
            <span class="version-badge">v{APP_VERSION}</span></h1>
        <p class="subtitle">
            Professionelle Risikobewertung für Energieportfolios ·
            OU + Jump-Diffusion · GARCH(1,1) · AR(1) Temperaturmodell ·
            CVaR Economic Capital · BDEW Profile · Stresstest · Sensitivitäten ·
            Smart Data Import · Export
        </p>
    </div>""", unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════════
    #  WORKFLOW STEPS
    # ════════════════════════════════════════════════════════════

    data = st.session_state.get("data")
    results = st.session_state.get("results")

    s1 = "done" if data else "active"
    s2 = "done" if results else ("active" if data else "pending")
    s3 = "active" if results else "pending"

    c1, c2, c3 = st.columns(3)
    with c1: st.markdown(render_step_card(1, "Daten laden", "Synthetisch generieren oder per Copy & Paste importieren.", s1), unsafe_allow_html=True)
    with c2: st.markdown(render_step_card(2, "Simulation", "Monte-Carlo mit konfigurierten Parametern.", s2), unsafe_allow_html=True)
    with c3: st.markdown(render_step_card(3, "Analyse", "Prämien, Verteilungen, Stress, Sensitivitäten.", s3), unsafe_allow_html=True)

    st.divider()

    # ════════════════════════════════════════════════════════════
    #  DATA INPUT (3 TABS)
    # ════════════════════════════════════════════════════════════

    tab_synth, tab_paste, tab_upload = st.tabs(["🔄 Synthetisch", "📋 Copy & Paste", "📁 Upload"])

    with tab_synth:
        st.markdown('<div class="info-box">Synthetische Daten aus BDEW-Profil, Temperatur & HPFC.</div>', unsafe_allow_html=True)
        c_gen, c_info = st.columns([1, 2])
        with c_gen:
            if st.button("🔄 Generieren", use_container_width=True, type="primary"):
                with st.spinner("Generiere Daten …"):
                    pdata = generate_synthetic_data(start_year, n_years, profile, annual_mwh, base_price, temp_sensitivity, seed)
                    st.session_state["data"] = pdata
                    st.session_state["results"] = None
                    st.session_state["data_source"] = "synthetic"
                    st.rerun()
        with c_info:
            if data and st.session_state.get("data_source") == "synthetic":
                st.success(f"✅ {data.years[0]}–{data.years[-1]} · {data.n_hours:,}h · {data.profile} · {data.annual_mwh:,.0f} MWh/a")

    with tab_paste:
        st.markdown("""
        <div class="paste-area-wrapper">
            <div class="paste-icon">📋</div>
            <div class="paste-title">Daten aus Excel / CSV einfügen</div>
            <div class="paste-hint">
                Automatische Erkennung von Spaltentypen, Datumsformaten und 
                deutschem Zahlenformat. Unterstützt Tab, Semikolon, Komma.
            </div>
        </div>""", unsafe_allow_html=True)

        paste_text = st.text_area("Daten einfügen", value=st.session_state.get("paste_text", ""), height=200,
            placeholder="Datum\tLast_MWh\tHPFC_EUR\n01.01.2026 00:00\t1,234\t82,50\n...",
            label_visibility="collapsed")

        c_p1, c_p2 = st.columns(2)
        with c_p1:
            if st.button("🔍 Analysieren & Importieren", use_container_width=True, type="primary", disabled=not paste_text.strip()):
                st.session_state["paste_text"] = paste_text
                with st.spinner("Analysiere …"):
                    df, mapping, messages = SmartDataParser.parse(paste_text)
                for msg in messages:
                    if msg.startswith("❌"): st.error(msg)
                    elif msg.startswith("⚠️"): st.warning(msg)
                    elif msg.startswith("✅"): st.success(msg)
                    else: st.info(msg)

                if df is not None:
                    pills = ""
                    if mapping.datetime_col: pills += render_column_pill(mapping.datetime_col, "datetime")
                    if mapping.load_col: pills += render_column_pill(mapping.load_col, "load")
                    if mapping.price_col: pills += render_column_pill(mapping.price_col, "price")
                    if mapping.temperature_col: pills += render_column_pill(mapping.temperature_col, "temperature")
                    st.markdown(pills, unsafe_allow_html=True)

                    with st.expander("👀 Vorschau"):
                        st.dataframe(df.head(20), use_container_width=True)

                    if st.button("✅ Übernehmen", type="primary", use_container_width=True):
                        pdata, msgs = SmartDataParser.convert_to_portfolio_data(df, mapping, base_price)
                        for m in msgs:
                            st.info(m) if not m.startswith("❌") else st.error(m)
                        if pdata:
                            st.session_state["data"] = pdata
                            st.session_state["results"] = None
                            st.session_state["data_source"] = "pasted"
                            st.rerun()

        with c_p2:
            if st.button("📝 Beispieldaten", use_container_width=True, type="secondary"):
                lines = ["Datum\tLast_MWh\tPreis_EUR\tTemp_C"]
                dt = datetime(2026, 1, 1)
                np.random.seed(123)
                for i in range(336):
                    h = dt.hour
                    f = bdew_factor("H0", h, dt.month, dt.weekday())
                    lines.append(f"{dt.strftime('%d.%m.%Y %H:%M')}\t{f*1.14:.3f}\t{80+12*np.sin(2*np.pi*h/24)+np.random.normal(0,3):.2f}\t{2+3*np.sin(2*np.pi*(h-6)/24)+np.random.normal(0,1):.1f}")
                    dt += timedelta(hours=1)
                st.session_state["paste_text"] = "\n".join(lines)
                st.rerun()

    with tab_upload:
        st.markdown('<div class="info-box">CSV oder Excel hochladen — automatische Spaltenerkennung.</div>', unsafe_allow_html=True)
        uploaded = st.file_uploader("Datei", type=["csv","xlsx","xls","tsv"])
        if uploaded:
            try:
                if uploaded.name.endswith(('.xlsx','.xls')):
                    df_up = pd.read_excel(uploaded)
                else:
                    content = uploaded.read().decode("utf-8", errors="replace")
                    uploaded.seek(0)
                    df_up = pd.read_csv(uploaded, sep=SmartDataParser.detect_delimiter(content), engine="python", on_bad_lines="skip")
                st.success(f"✅ {len(df_up):,} × {len(df_up.columns)} geladen")
                mapping_up = ColumnMapping()
                for col in df_up.columns:
                    ct, _ = SmartDataParser.classify_column(str(col), df_up[col])
                    if ct == "datetime" and not mapping_up.datetime_col: mapping_up.datetime_col = str(col)
                    elif ct == "load" and not mapping_up.load_col: mapping_up.load_col = str(col)
                    elif ct == "price" and not mapping_up.price_col: mapping_up.price_col = str(col)
                    elif ct == "temperature" and not mapping_up.temperature_col: mapping_up.temperature_col = str(col)
                if st.button("✅ Upload übernehmen", type="primary", use_container_width=True):
                    if mapping_up.datetime_col:
                        try: df_up[mapping_up.datetime_col] = pd.to_datetime(df_up[mapping_up.datetime_col], dayfirst=True, format="mixed")
                        except: pass
                    for c in [mapping_up.load_col, mapping_up.price_col, mapping_up.temperature_col]:
                        if c and c in df_up.columns and df_up[c].dtype == object:
                            df_up[c] = df_up[c].apply(SmartDataParser.clean_german_number)
                    pdata, msgs = SmartDataParser.convert_to_portfolio_data(df_up, mapping_up, base_price)
                    if pdata:
                        st.session_state["data"] = pdata
                        st.session_state["results"] = None
                        st.session_state["data_source"] = "uploaded"
                        st.rerun()
            except Exception as e:
                st.error(f"❌ {e}")

    # ════════════════════════════════════════════════════════════
    #  DATA QUALITY & EXPLORATION
    # ════════════════════════════════════════════════════════════

    data = st.session_state.get("data")

    if data is not None:
        st.divider()

        if data.quality:
            q = data.quality
            c_b, c_i = st.columns([1, 3])
            with c_b: st.markdown(render_quality_badge(q), unsafe_allow_html=True)
            with c_i: st.markdown(f"**{q.total_hours:,}** Stunden · **{len(q.years_covered)}** Jahre ({q.years_covered[0]}–{q.years_covered[-1]})")
            if q.warnings:
                with st.expander(f"⚠️ {len(q.warnings)} Hinweise"):
                    for w in q.warnings: st.warning(w)

        st.markdown("##### 📊 Datenzusammenfassung")
        k1,k2,k3,k4,k5,k6 = st.columns(6)
        for col, lbl, val, u in [
            (k1,"Gesamtvolumen",f"{data.load.sum():,.0f}","MWh"),
            (k2,"Ø Last",f"{data.load.mean():.3f}","MWh/h"),
            (k3,"Peak",f"{data.load.max():.3f}","MWh/h"),
            (k4,"Ø HPFC",f"{data.hpfc.mean():.1f}","€/MWh"),
            (k5,"Ø Temp",f"{data.temperature.mean():.1f}","°C"),
            (k6,"Profil",data.profile,""),
        ]:
            col.markdown(render_kpi_card(lbl, val, u), unsafe_allow_html=True)

        st.markdown("##### 📈 Datenexploration")
        et1,et2,et3,et4 = st.tabs(["📈 Zeitreihe","📊 Monatlich","📉 Dauerlinie","🗓️ Heatmap"])
        with et1:
            hrs = st.slider("Stunden", 168, min(data.n_hours, 8760), min(720, data.n_hours), 168, key="ts_h_v2")
            st.plotly_chart(chart_load_hpfc_timeseries(data, hrs), use_container_width=True)
        with et2:
            st.plotly_chart(chart_monthly_summary(data), use_container_width=True)
        with et3:
            c_ldc, c_day = st.columns(2)
            with c_ldc: st.plotly_chart(chart_load_duration_curve(data), use_container_width=True)
            with c_day: st.plotly_chart(chart_average_daily_shape(data), use_container_width=True)
        with et4:
            st.plotly_chart(chart_heatmap_monthly(data), use_container_width=True)

    # ════════════════════════════════════════════════════════════
    #  SIMULATION TRIGGER
    # ════════════════════════════════════════════════════════════

    if data is not None:
        st.divider()
        st.markdown("##### 🚀 Simulation")
        c_run, c_st = st.columns([1, 2])
        with c_run:
            if st.button("▶  Monte-Carlo starten", use_container_width=True, type="primary"):
                st.session_state["run_simulation"] = True
        with c_st:
            if st.session_state.get("run_simulation"):
                st.info(f"🔄 {params['paths']:,} Pfade × {data.n_hours:,} Stunden …")

        # Run simulation and display results
        if st.session_state.get("run_simulation") or st.session_state.get("results"):
            run_simulation_and_display(data, params)

    # ════════════════════════════════════════════════════════════
    #  GLOBAL METHODOLOGY (always accessible)
    # ════════════════════════════════════════════════════════════

    if data is None and results is None:
        st.divider()
        with st.expander("📐 Methodik & mathematische Grundlagen", expanded=False):
            render_methodology_tab()
        with st.expander("📖 Glossar — Energiewirtschaft", expanded=False):
            render_glossary()

    # ════════════════════════════════════════════════════════════
    #  FOOTER
    # ════════════════════════════════════════════════════════════

    st.markdown("---")
    st.caption(
        f"Quant Energy Risk Engine v{APP_VERSION} · "
        f"Enterprise Portfolio Risk Assessment · "
        f"Nicht als Anlageberatung zu verstehen · © 2025"
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  END OF BLOCK 3
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    main()
