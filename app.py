"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                   QUANTITATIVE ENERGY RISK ENGINE  v3.0                     ║
║          Multi-Year Portfolio Assessment via Monte Carlo Simulation          ║
║                                                                              ║
║  Spot Price   :  Ornstein-Uhlenbeck  +  Merton Jump-Diffusion               ║
║  Volume Error :  Autoregressive AR(1) with Weather Persistence               ║
║  Dependence   :  Cholesky-decomposed Bivariate Normal Innovations            ║
║  Risk Capital :  CVaR-based Economic Capital with RORAC Pricing              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import time

# ═══════════════════════════════════════════════════════════════════════════
# 1 ── PAGE CONFIG & CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Quant Energy Risk Engine",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

COL = dict(
    blue="#3b82f6", indigo="#6366f1", emerald="#10b981",
    amber="#f59e0b", rose="#ef4444", cyan="#06b6d4",
    slate="#94a3b8", white="#f8fafc",
)

_PLT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(15,23,42,0.4)",
    font=dict(family="Inter, system-ui, sans-serif", color="#cbd5e1", size=12),
    margin=dict(l=55, r=30, t=55, b=45),
    hoverlabel=dict(font_size=12),
)


# ═══════════════════════════════════════════════════════════════════════════
# 2 ── CUSTOM CSS
# ═══════════════════════════════════════════════════════════════════════════

st.markdown("""
<style>
    /* ── layout ── */
    .block-container{padding-top:1.5rem;max-width:1480px}
    header[data-testid="stHeader"]{background:rgba(15,23,42,.85);backdrop-filter:blur(12px)}

    /* ── hero card ── */
    .hero{
        background:linear-gradient(135deg,#1e3a5f 0%,#0f172a 100%);
        border:1px solid #1e40af;border-radius:16px;padding:2rem 2.5rem;
        margin-bottom:1.2rem;
    }
    .hero h1{margin:0;color:#f8fafc;font-size:1.65rem;font-weight:800;letter-spacing:-.025em}
    .hero p{margin:.4rem 0 0;color:#93c5fd;font-size:.82rem}

    /* ── KPI strip ── */
    .kpi{
        background:linear-gradient(145deg,#1e293b 0%,#0f172a 100%);
        border:1px solid #334155;border-radius:14px;padding:1.3rem 1rem;text-align:center;
    }
    .kpi .l{font-size:.68rem;color:#94a3b8;text-transform:uppercase;letter-spacing:.09em;margin-bottom:.35rem}
    .kpi .v{font-size:1.75rem;font-weight:800;color:#f8fafc}
    .kpi .u{font-size:.78rem;color:#64748b;margin-left:2px}

    /* ── big result ── */
    .bigres{
        background:linear-gradient(135deg,#172554 0%,#0f172a 100%);
        border:1px solid rgba(59,130,246,.45);border-radius:18px;
        padding:2.2rem;text-align:center;
    }
    .bigres .l{font-size:.72rem;color:#93c5fd;text-transform:uppercase;letter-spacing:.12em}
    .bigres .v{font-size:3rem;font-weight:900;color:#fff;letter-spacing:-.03em}

    /* ── table ── */
    .rtbl{width:100%;border-collapse:collapse;font-size:.84rem}
    .rtbl th{
        background:#0f172a;color:#94a3b8;padding:11px 14px;text-align:left;
        font-size:.7rem;text-transform:uppercase;letter-spacing:.06em;
        border-bottom:2px solid #1e40af;font-weight:600;
    }
    .rtbl td{padding:11px 14px;border-bottom:1px solid #1e293b;color:#e2e8f0}
    .rtbl tr:hover td{background:rgba(30,41,59,.5)}
    .rtbl .tot td{background:#0f172a;font-weight:700;border-top:2px solid #3b82f6}
    .rtbl .mono{font-family:'JetBrains Mono',monospace;font-size:.82rem}
    .rtbl .r{text-align:right}
    .rtbl .sub{font-size:.62rem;color:#64748b;margin-top:2px}

    /* ── methodology ── */
    .methbox{
        background:#0f172a;border:1px solid #1e293b;border-radius:12px;
        padding:1.6rem 2rem;margin-bottom:1rem;
    }

    /* ── misc ── */
    div[data-testid="stExpander"] details{border-color:#334155!important;border-radius:12px!important}
    section[data-testid="stSidebar"]{background:#0f172a}
    section[data-testid="stSidebar"] .stMarkdown h3{color:#e2e8f0;font-size:1rem}
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# 3 ── BDEW STANDARD LOAD PROFILE GENERATOR
# ═══════════════════════════════════════════════════════════════════════════

def bdew_factor(profile: str, hour: int, month: int, weekday: int) -> float:
    """
    Synthetic BDEW-type load-profile factor.

    Parameters
    ----------
    profile  : 'H0' | 'G0' | 'G1' | 'RLM'
    hour     : 0-23
    month    : 1-12
    weekday  : 0=Mon … 6=Sun

    Returns
    -------
    Dimensionless scaling factor (mean ≈ 1 before normalisation).
    """
    wknd = weekday >= 5
    winter = month in (1, 2, 3, 11, 12)
    summer = month in (6, 7, 8)

    if profile == "H0":
        s = 1.35 if winter else (0.65 if summer else 1.0)
        if wknd:
            h = 1.15 if 8 <= hour <= 12 else (1.25 if 13 <= hour <= 22 else (0.9 if hour == 7 else 0.55))
        else:
            if   6 <= hour <= 8:  h = 1.25
            elif 9 <= hour <= 12: h = 0.95
            elif 13 <= hour <= 16:h = 0.85
            elif 17 <= hour <= 21:h = 1.75
            elif hour in (22, 23):h = 1.10
            else:                 h = 0.45
        return s * h

    if profile == "G0":
        s = 1.10 if winter else (0.85 if summer else 0.95)
        if wknd:
            return s * (0.35 if 6 <= hour <= 20 else 0.25)
        if   7 <= hour <= 12: return s * 1.60
        if  13 <= hour <= 17: return s * 1.45
        if   6 <= hour <= 7:  return s * 0.90
        if  17 <= hour <= 19: return s * 0.80
        return s * 0.30

    if profile == "G1":
        s = 1.05 if winter else 0.95
        if wknd: return s * 0.15
        if 8 <= hour <= 18: return s * 1.70
        if 6 <= hour <= 8 or 18 <= hour <= 20: return s * 0.60
        return s * 0.20

    # RLM / Bandlast
    return 1.0 + 0.08 * np.sin(2 * np.pi * hour / 24) + 0.05 * np.cos(2 * np.pi * (month - 1) / 12)


# ═══════════════════════════════════════════════════════════════════════════
# 4 ── SYNTHETIC DATA GENERATOR
# ═══════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def synthesise_data(
    start_year: int, n_years: int, profile: str,
    annual_mwh: float, base_price: float, _seed: int = 42,
) -> dict:
    """Build hourly load & HPFC arrays for *n_years* starting at *start_year*."""
    np.random.seed(_seed)
    start = datetime(start_year, 1, 1)
    end   = datetime(start_year + n_years, 1, 1)

    dates, loads, hpfcs, ymap = [], [], [], []
    years_found: list[int] = []
    cur = start
    yidx = -1; last_y = -1; h = 0

    while cur < end:
        y = cur.year
        if y != last_y:
            years_found.append(y); yidx += 1; last_y = y
        dates.append(cur)
        ymap.append(yidx)
        loads.append(bdew_factor(profile, cur.hour, cur.month, cur.weekday()))
        contango  = yidx * 2.5
        intraday  = 12.0 * np.sin(2 * np.pi * cur.hour / 24 - np.pi / 3)
        seasonal  = 10.0 * np.cos(2 * np.pi * (cur.month - 1) / 12)
        hpfcs.append(base_price + contango + intraday + seasonal + np.random.normal(0, 1.2))
        cur += timedelta(hours=1); h += 1

    load_arr = np.array(loads, dtype=np.float64)
    ym       = np.array(ymap, dtype=np.int32)
    for yi in range(n_years):
        mask = ym == yi
        s = load_arr[mask].sum()
        if s > 0:
            load_arr[mask] *= annual_mwh / s

    return dict(
        dates=dates, load=load_arr, hpfc=np.array(hpfcs),
        year_map=ym, years=years_found, n_hours=h,
    )


# ═══════════════════════════════════════════════════════════════════════════
# 5 ── RISK-METRIC CALCULATOR
# ═══════════════════════════════════════════════════════════════════════════

def risk_metrics(losses: np.ndarray, alpha: float = 0.95) -> dict:
    """
    Compute EL, VaR, CVaR, UL, σ, skewness, excess kurtosis
    from a 1-D loss sample.
    """
    n   = len(losses)
    el  = losses.mean()
    std = losses.std(ddof=1) if n > 1 else 0.0
    s   = np.sort(losses)
    idx = int(np.floor(alpha * n))
    var = s[min(idx, n - 1)]
    tail = s[idx:]
    cvar = tail.mean() if len(tail) else var
    ul   = max(0.0, cvar - el)
    if std > 1e-12:
        z   = (losses - el) / std
        skw = (z ** 3).mean()
        krt = (z ** 4).mean() - 3.0
    else:
        skw = krt = 0.0
    return dict(EL=el, VaR=var, CVaR=cvar, UL=ul,
                std=std, skew=skw, kurtosis=krt, sorted=s)


# ═══════════════════════════════════════════════════════════════════════════
# 6 ── MONTE-CARLO SIMULATION ENGINE  (numpy-vectorised)
# ═══════════════════════════════════════════════════════════════════════════

def run_mc(data: dict, params: dict) -> dict:
    """
    Vectorised Monte-Carlo engine.

    For every hourly step *t* the following are computed **simultaneously
    across all N paths** using pure NumPy operations:

    1.  **Spot price** via Ornstein–Uhlenbeck + Merton Jump-Diffusion
    2.  **Volume error** via AR(1) with weather-persistence parameter φ
    3.  **Imbalance cost** (reBAP penalty model – always adverse)
    4.  **Volume-price risk** (two-sided ΔQ × (Spot − HPFC))

    Returns a nested dict with yearly + portfolio-level risk decomposition.
    """
    hpfc     = data["hpfc"]
    load     = data["load"]
    ym       = data["year_map"]
    n_years  = len(data["years"])
    T        = len(load)
    N        = params["paths"]

    # ── model parameters ──
    kappa     = params["kappa"]
    sigma_p   = params["sigma_price"]
    sigma_v   = params["vol_error"] / 100.0
    phi       = params["phi"]
    rho       = params["correlation"] / 100.0
    lam       = params["jump_prob"]
    sigma_j   = params["jump_size"]
    r_ec      = params["cost_of_capital"] / 100.0
    alpha     = params["confidence"]
    bp        = params["base_price"]

    L11       = np.sqrt(max(1e-12, 1.0 - rho ** 2))
    phi_f     = np.sqrt(max(1e-12, 1.0 - phi ** 2))

    # ── accumulators ──
    imb  = np.zeros((N, n_years), dtype=np.float64)
    vp   = np.zeros((N, n_years), dtype=np.float64)
    vol_y = np.zeros(n_years); pc_y = np.zeros(n_years)
    for t in range(T):
        vol_y[ym[t]] += load[t]
        pc_y[ym[t]]  += hpfc[t] * load[t]

    # ── state vectors (N paths) ──
    spot = np.full(N, hpfc[0])
    vs   = np.zeros(N)

    bar  = st.progress(0, text="Initialising stochastic paths …")
    step = max(1, T // 200)

    for t in range(T):
        y = ym[t]

        # correlated normals
        z1 = np.random.standard_normal(N)
        z2 = np.random.standard_normal(N)
        dw_p = z1
        dw_v = rho * z1 + L11 * z2

        # jump component  (Poisson arrivals)
        jm = (np.random.random(N) < lam).astype(np.float64)
        jumps = jm * np.random.normal(0, sigma_j, N)

        # OU spot dynamics
        spot += kappa * (hpfc[t] - spot) + sigma_p * dw_p + jumps
        np.clip(spot, -100, 1500, out=spot)

        # AR(1) volume error
        vs[:] = phi * vs + phi_f * sigma_v * dw_v
        act   = load[t] * (1.0 + vs)
        np.maximum(act, 0.0, out=act)

        de = act - load[t]

        # reBAP penalty (half-normal ≥ 5 €/MWh)
        pen = 5.0 + np.abs(np.random.standard_normal(N)) * 15.0
        imb[:, y] += np.abs(de) * pen
        vp[:, y]  += de * (spot - hpfc[t])

        if t % step == 0:
            bar.progress(t / T, text=f"t = {t:,} / {T:,}  ·  {N:,} paths")

    bar.progress(1.0, text="✓  Simulation complete"); time.sleep(.35); bar.empty()

    # ── per-year evaluation ──
    yearly = []
    for yi in range(n_years):
        V = vol_y[yi]
        sp = (pc_y[yi] / V - bp) if V else 0.0
        im = risk_metrics(imb[:, yi], alpha)
        vm = risk_metrics(vp[:, yi],  alpha)
        pp = (im["EL"] + im["UL"] * r_ec) / V if V else 0.0
        vpp= (vm["EL"] + vm["UL"] * r_ec) / V if V else 0.0
        yearly.append(dict(
            yi=yi, vol=V, struktur=sp,
            prognose=pp, vp_prem=vpp,
            gesamt=sp + pp + vpp,
            im=im, vm=vm,
        ))

    # ── portfolio level (diversification) ──
    imb_tot = imb.sum(axis=1)
    vp_tot  = vp.sum(axis=1)
    Vt = vol_y.sum()
    sp_t = (pc_y.sum() / Vt - bp)
    im_t = risk_metrics(imb_tot, alpha)
    vm_t = risk_metrics(vp_tot, alpha)
    pp_t = (im_t["EL"] + im_t["UL"] * r_ec) / Vt
    vp_t = (vm_t["EL"] + vm_t["UL"] * r_ec) / Vt
    g_t  = sp_t + pp_t + vp_t

    # ── diversification benefit ──
    sum_standalone = sum(
        (yr["prognose"] + yr["vp_prem"]) * yr["vol"] for yr in yearly
    ) / Vt
    portfolio_risk = pp_t + vp_t
    div_benefit = sum_standalone - portfolio_risk

    return dict(
        yearly=yearly,
        total=dict(vol=Vt, struktur=sp_t, prognose=pp_t, vp_prem=vp_t,
                   gesamt=g_t, im=im_t, vm=vm_t, div_benefit=div_benefit),
        raw=dict(imb_y=imb, vp_y=vp, imb_tot=imb_tot, vp_tot=vp_tot),
    )


# ═══════════════════════════════════════════════════════════════════════════
# 7 ── PLOTLY CHART BUILDERS
# ═══════════════════════════════════════════════════════════════════════════

def _layout(**kw):
    d = {**_PLT}
    d.update(kw)
    return d


def chart_waterfall(res):
    vals = [res["total"]["struktur"], res["total"]["prognose"],
            res["total"]["vp_prem"], res["total"]["gesamt"]]
    fig = go.Figure(go.Waterfall(
        orientation="v",
        measure=["relative", "relative", "relative", "total"],
        x=["Strukturbeitrag", "Prognoserisiko", "VP-Risiko", "Gesamtaufschlag"],
        y=vals,
        text=[f"{v:+.2f}" for v in vals],
        textposition="outside",
        textfont=dict(family="JetBrains Mono", size=13, color="#e2e8f0"),
        connector=dict(line=dict(color="#334155", width=1.5, dash="dot")),
        increasing=dict(marker=dict(color=COL["rose"], line=dict(color=COL["rose"], width=1))),
        decreasing=dict(marker=dict(color=COL["emerald"], line=dict(color=COL["emerald"], width=1))),
        totals=dict(marker=dict(color=COL["blue"], line=dict(color=COL["blue"], width=1))),
    ))
    fig.update_layout(**_layout(
        title=dict(text="Risk-Premium Decomposition  (€ / MWh)", font=dict(size=15)),
        yaxis_title="€ / MWh", showlegend=False, height=420,
    ))
    return fig


def chart_annual_bars(res, years):
    yr = res["yearly"]
    x  = [str(y) for y in years]
    fig = go.Figure()
    for key, name, col in [
        ("struktur",  "Struktur",    COL["indigo"]),
        ("prognose",  "Prognose",    COL["amber"]),
        ("vp_prem",   "VP-Risiko",   COL["rose"]),
    ]:
        fig.add_trace(go.Bar(x=x, y=[r[key] for r in yr], name=name,
                             marker_color=col, marker_line=dict(width=0)))
    fig.update_layout(**_layout(
        barmode="stack",
        title=dict(text="Annual Risk-Premium Stack  (€ / MWh)", font=dict(size=15)),
        yaxis_title="€ / MWh", height=400,
        legend=dict(orientation="h", y=1.08, x=.5, xanchor="center",
                    font=dict(size=11)),
    ))
    return fig


def chart_distribution(losses, label, colour, alpha=0.95):
    m = risk_metrics(losses, alpha)
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=losses, nbinsx=120, name=label,
        marker_color=f"rgba({_hex2rgb(colour)},.55)",
        marker_line=dict(color=colour, width=.8),
    ))
    for val, nm, c, dash in [
        (m["EL"],   "E[L]",           COL["emerald"], "dot"),
        (m["VaR"],  f"VaR{int(alpha*100)}",  COL["amber"],  "dash"),
        (m["CVaR"], f"CVaR{int(alpha*100)}", COL["rose"],   "solid"),
    ]:
        fig.add_vline(x=val, line=dict(color=c, width=2, dash=dash))
        fig.add_annotation(x=val, y=1, yref="paper", text=f"  {nm}={val:,.0f}€",
                           showarrow=False, font=dict(color=c, size=11),
                           xanchor="left", yanchor="top")
    fig.update_layout(**_layout(
        title=dict(text=f"Portfolio {label} Distribution  (€ absolute)", font=dict(size=14)),
        xaxis_title="€", yaxis_title="Frequency", showlegend=False, height=370,
    ))
    return fig


def chart_profile(data, max_hours=504):
    """First week (or max_hours) of load + HPFC."""
    n = min(max_hours, data["n_hours"])
    x = data["dates"][:n]
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(
        x=x, y=data["load"][:n], name="Last (MWh)",
        line=dict(color=COL["cyan"], width=1.4), fill="tozeroy",
        fillcolor="rgba(6,182,212,.12)"), secondary_y=False)
    fig.add_trace(go.Scatter(
        x=x, y=data["hpfc"][:n], name="HPFC (€/MWh)",
        line=dict(color=COL["amber"], width=1.6, dash="dot")),
        secondary_y=True)
    fig.update_layout(**_layout(
        title=dict(text=f"Load Profile & HPFC  (first {n} h)", font=dict(size=14)),
        height=360, legend=dict(orientation="h", y=1.10, x=.5, xanchor="center"),
    ))
    fig.update_yaxes(title_text="MWh", secondary_y=False, gridcolor="rgba(51,65,85,.3)")
    fig.update_yaxes(title_text="€/MWh", secondary_y=True, gridcolor="rgba(51,65,85,.15)")
    return fig


def chart_convergence(losses, alpha=0.95, steps=80):
    """CVaR convergence as N grows."""
    n = len(losses)
    ns = np.unique(np.logspace(np.log10(50), np.log10(n), steps).astype(int))
    cvars = [risk_metrics(losses[:k], alpha)["CVaR"] for k in ns]
    fig = go.Figure(go.Scatter(
        x=ns, y=cvars, mode="lines+markers",
        marker=dict(size=3, color=COL["blue"]),
        line=dict(color=COL["blue"], width=1.6), name="CVaR"))
    fig.add_hline(y=cvars[-1], line=dict(color=COL["slate"], dash="dash", width=1),
                  annotation_text=f"final = {cvars[-1]:,.0f}€")
    fig.update_layout(**_layout(
        title=dict(text="CVaR Convergence Diagnostic", font=dict(size=14)),
        xaxis_title="Number of paths used", yaxis_title="CVaR (€)",
        height=340, showlegend=False,
    ))
    return fig


def _hex2rgb(h):
    h = h.lstrip("#")
    return ",".join(str(int(h[i:i+2], 16)) for i in (0, 2, 4))


# ═══════════════════════════════════════════════════════════════════════════
# 8 ── METHODOLOGY (LaTeX)
# ═══════════════════════════════════════════════════════════════════════════

METHODOLOGY_MD = r"""
<div class="methbox">

### 1 &nbsp; Spot-Price Dynamics — Ornstein-Uhlenbeck + Merton Jump-Diffusion

$$
dS_t \;=\; \kappa\bigl(\mu_t - S_t\bigr)\,dt
       \;+\; \sigma_S\,dW_t^{(1)}
       \;+\; J_t\,dN_t
$$

| Symbol | Meaning |
|--------|---------|
| $\mu_t$ | HPFC forward curve (deterministic mean) |
| $\kappa$ | Mean-reversion speed  (h⁻¹) |
| $\sigma_S$ | Diffusion volatility  (€ / MWh) |
| $J_t \sim \mathcal{N}(0,\,\sigma_J^2)$ | Jump size |
| $N_t \sim \text{Poisson}(\lambda)$ | Jump arrival process |

</div>

<div class="methbox">

### 2 &nbsp; Volume-Error Dynamics — AR(1) with Weather Persistence

$$
\varepsilon_t \;=\; \varphi\,\varepsilon_{t-1}
               \;+\; \sqrt{1 - \varphi^2}\;\sigma_V\;dW_t^{(2)}
$$

$$
Q_t^{\text{act}} \;=\; Q_t^{\text{fcst}} \cdot \bigl(1 + \varepsilon_t\bigr)
$$

The parameter $\varphi \in [0,1)$ controls **weather persistence**:
a high value means cold or warm spells persist across many hours,
creating sustained deviations from the forecast profile and
therefore structural portfolio exposure.

</div>

<div class="methbox">

### 3 &nbsp; Bivariate Dependence — Cholesky Decomposition

$$
dW_t^{(2)} \;=\; \rho\;dW_t^{(1)}
            \;+\; \sqrt{1-\rho^2}\;dZ_t
$$

where $\rho$ is the instantaneous **price-volume correlation**.
A positive $\rho$ (empirically ≈ 0.3 – 0.5 in Central Europe)
implies that cold weather simultaneously increases demand **and**
spot prices, amplifying portfolio risk.

</div>

<div class="methbox">

### 4 &nbsp; Risk Quantification — CVaR Economic Capital

$$
\text{VaR}_\alpha(L) \;=\; \inf\bigl\{x : \Pr(L \le x) \ge \alpha\bigr\}
$$

$$
\text{CVaR}_\alpha(L) \;=\; \mathbb{E}\bigl[L \;\big|\; L > \text{VaR}_\alpha(L)\bigr]
$$

$$
\text{UL} \;=\; \max\!\bigl(0,\;\text{CVaR}_\alpha - \mathbb{E}[L]\bigr)
$$

$$
\text{Premium} \;=\;
\frac{
  \underbrace{\mathbb{E}[L]}_{\text{Expected Loss}}
  \;+\;
  \underbrace{r_{EC} \cdot \text{UL}}_{\text{Cost of Risk Capital}}
}{
  \displaystyle\sum_t V_t
}
$$

</div>

<div class="methbox">

### 5 &nbsp; Diversification Benefit

Because $\text{CVaR}$ is **sub-additive** for elliptical distributions:

$$
\text{CVaR}\!\Bigl(\sum_{y} L_y\Bigr)
\;\le\;
\sum_{y} \text{CVaR}(L_y)
$$

The **portfolio premium** over multiple delivery years is therefore
strictly less than the volume-weighted sum of standalone year premiums.
This difference is reported as the *Diversification Benefit*.

</div>
"""


# ═══════════════════════════════════════════════════════════════════════════
# 9 ── MAIN APPLICATION
# ═══════════════════════════════════════════════════════════════════════════

def main():

    # ── session state defaults ──
    for k, v in [("data", None), ("results", None)]:
        if k not in st.session_state:
            st.session_state[k] = v

    # ────────────────────────── SIDEBAR ──────────────────────────

    with st.sidebar:
        st.markdown("### ⚡ &nbsp;Engine Configuration")

        with st.expander("📐  BDEW Profile Synthesis", expanded=True):
            profile = st.selectbox("Profile type",
                                   ["H0", "G0", "G1", "RLM"],
                                   format_func=lambda x: {
                                       "H0": "H0  —  Household",
                                       "G0": "G0  —  Commercial",
                                       "G1": "G1  —  Comm. 8-18",
                                       "RLM":"RLM —  Baseload",
                                   }[x])
            annual_mwh = st.number_input("Annual volume  (MWh)", 500, 500_000, 10_000, step=500)
            c1, c2 = st.columns(2)
            start_year = c1.number_input("Start year", 2025, 2035, 2026)
            n_years    = c2.selectbox("Duration", [1,2,3,4,5], index=2,
                                      format_func=lambda n: f"{n} yr{'s' if n>1 else ''}")

        with st.expander("📊  Market & Capital Parameters", expanded=True):
            base_price = st.number_input("Base price — front year  (€/MWh)",
                                         10.0, 300.0, 80.0, step=5.0, format="%.1f")
            c1, c2 = st.columns(2)
            confidence = c1.selectbox("Confidence α", [0.90, 0.95, 0.975, 0.99],
                                      index=1, format_func=lambda x: f"{x:.1%}")
            cost_cap   = c2.number_input("Cost of equity  (%)", 5.0, 30.0, 15.0,
                                          step=1.0, format="%.1f")

        with st.expander("🔬  Stochastic Calibration", expanded=False):
            paths      = st.select_slider("MC paths", [500,1000,2000,5000,10000], 2000)
            vol_error  = st.slider("Volume-error  σ_V  (%)", 2.0, 20.0, 8.0, 0.5)
            corr       = st.slider("Correlation  ρ  (%)", -50.0, 80.0, 40.0, 5.0)
            kappa_ui   = st.slider("Mean reversion  κ", 0.01, 0.50, 0.10, 0.01)
            phi_ui     = st.slider("AR(1) persistence  φ", 0.50, 0.995, 0.95, 0.005,
                                    format="%.3f")
            sig_p      = st.slider("Price diffusion  σ_S  (€)", 5.0, 40.0, 15.0, 1.0)
            j_prob     = st.slider("Jump probability  λ  (per h)", 0.0, 0.10, 0.02, 0.005,
                                    format="%.3f")
            j_size     = st.slider("Jump volatility  σ_J  (€)", 10.0, 200.0, 80.0, 10.0)
            seed       = st.number_input("Random seed", 0, 99999, 42, step=1)

    params = dict(
        base_price=base_price, paths=paths, confidence=confidence,
        cost_of_capital=cost_cap, vol_error=vol_error,
        correlation=corr, kappa=kappa_ui, phi=phi_ui,
        sigma_price=sig_p, jump_prob=j_prob, jump_size=j_size,
    )

    # ────────────────────────── HEADER ──────────────────────────

    st.markdown("""
    <div class="hero">
        <h1>⚡&nbsp; Quantitative Energy Risk Engine &nbsp;<span style="font-weight:400;
            font-size:.85rem;color:#60a5fa">v3.0</span></h1>
        <p>Ornstein-Uhlenbeck + Jump-Diffusion  ·  AR(1) Volume  ·
           CVaR Economic Capital  ·  Multi-Year Diversification</p>
    </div>""", unsafe_allow_html=True)

    # ────────────────────────── CONTROLS ──────────────────────────

    col_gen, col_run, col_status = st.columns([1, 1, 2])

    with col_gen:
        if st.button("🔄  Generate Synthetic Data", use_container_width=True,
                      type="secondary"):
            with st.spinner("Building BDEW profiles …"):
                d = synthesise_data(start_year, n_years, profile,
                                    annual_mwh, base_price, _seed=seed)
                st.session_state.data = d
                st.session_state.results = None

    with col_run:
        run_disabled = st.session_state.data is None
        if st.button("▶  Run Monte-Carlo", use_container_width=True,
                      type="primary", disabled=run_disabled):
            np.random.seed(seed)
            res = run_mc(st.session_state.data, params)
            st.session_state.results = res

    with col_status:
        d = st.session_state.data
        if d is not None:
            yrs = d["years"]
            st.success(
                f"**{yrs[0]}–{yrs[-1]}**  ·  {d['n_hours']:,} hours  ·  "
                f"Profile {profile}  ·  {annual_mwh:,.0f} MWh / a",
                icon="📅",
            )
        else:
            st.info("Generate data first, then run the simulation.", icon="💡")

    st.divider()

    # ────────────────────────── RESULTS ──────────────────────────

    res = st.session_state.results
    if res is None:
        # ── show profile preview if data exists ──
        if st.session_state.data is not None:
            st.plotly_chart(chart_profile(st.session_state.data), use_container_width=True)
        return

    tot  = res["total"]
    yrs  = st.session_state.data["years"]

    # ── KPI STRIP ──
    k1, k2, k3, k4, k5 = st.columns(5)
    for c, label, val, fmt, unit in [
        (k1, "Strukturbeitrag", tot["struktur"],  "{:+.2f}",  "€/MWh"),
        (k2, "Prognoserisiko",  tot["prognose"],  "+{:.2f}",  "€/MWh"),
        (k3, "VP-Risiko",       tot["vp_prem"],   "+{:.2f}",  "€/MWh"),
        (k4, "Diversifikation", tot["div_benefit"],"−{:.2f}",  "€/MWh"),
        (k5, "Gesamtvolumen",   tot["vol"],       "{:,.0f}",  "MWh"),
    ]:
        c.markdown(
            f'<div class="kpi"><div class="l">{label}</div>'
            f'<div class="v">{fmt.format(val)}<span class="u">{unit}</span></div></div>',
            unsafe_allow_html=True,
        )

    st.markdown("")

    # ── BIG RESULT ──
    st.markdown(
        f'<div class="bigres"><div class="l">Portfolio-Mischpreis — Gesamtaufschlag</div>'
        f'<div class="v">{tot["gesamt"]:+.2f} <span style="font-size:1.2rem;'
        f'color:#60a5fa;font-weight:500">€ / MWh</span></div></div>',
        unsafe_allow_html=True,
    )

    st.markdown("")

    # ── TABS ──
    tab1, tab2, tab3, tab4 = st.tabs([
        "📋  Granular Results",
        "📈  Risk Analytics",
        "📊  Data Profile",
        "📐  Methodology",
    ])

    # ╌╌╌╌╌╌╌╌╌╌  TAB 1 — TABLE + WATERFALL  ╌╌╌╌╌╌╌╌╌╌
    with tab1:
        c_table, c_wf = st.columns([3, 2])

        with c_table:
            rows_html = ""
            for yr in res["yearly"]:
                y_label = str(yrs[yr["yi"]])
                rows_html += f"""<tr>
                    <td style="font-weight:700">{y_label}</td>
                    <td class="mono r">{yr['vol']:,.0f}</td>
                    <td class="mono r" style="color:#a5b4fc">{yr['struktur']:+.2f}</td>
                    <td class="mono r" style="color:#fbbf24">+{yr['prognose']:.2f}
                        <div class="sub">EL {yr['im']['EL']/yr['vol']:.2f}
                          · UL {yr['im']['UL']/yr['vol']:.2f}</div></td>
                    <td class="mono r" style="color:#fb7185">+{yr['vp_prem']:.2f}
                        <div class="sub">EL {yr['vm']['EL']/yr['vol']:.2f}
                          · UL {yr['vm']['UL']/yr['vol']:.2f}</div></td>
                    <td class="mono r" style="font-weight:700">{yr['gesamt']:+.2f}</td>
                </tr>"""

            rows_html += f"""<tr class="tot">
                <td>Portfolio</td>
                <td class="mono r">{tot['vol']:,.0f}</td>
                <td class="mono r" style="color:#818cf8">{tot['struktur']:+.2f}</td>
                <td class="mono r" style="color:#fbbf24">+{tot['prognose']:.2f}</td>
                <td class="mono r" style="color:#fb7185">+{tot['vp_prem']:.2f}</td>
                <td class="mono r" style="color:#60a5fa;font-size:1.05rem">
                    {tot['gesamt']:+.2f}</td>
            </tr>"""

            st.markdown(f"""
            <table class="rtbl">
            <thead><tr>
                <th>Year</th><th style="text-align:right">Volume</th>
                <th style="text-align:right">Struktur</th>
                <th style="text-align:right">Prognose</th>
                <th style="text-align:right">VP-Risk</th>
                <th style="text-align:right">Total</th>
            </tr></thead>
            <tbody>{rows_html}</tbody>
            </table>""", unsafe_allow_html=True)

        with c_wf:
            st.plotly_chart(chart_waterfall(res), use_container_width=True)

        st.plotly_chart(chart_annual_bars(res, yrs), use_container_width=True)

    # ╌╌╌╌╌╌╌╌╌╌  TAB 2 — DISTRIBUTIONS  ╌╌╌╌╌╌╌╌╌╌
    with tab2:
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(
                chart_distribution(res["raw"]["imb_tot"], "Imbalance Cost",
                                   COL["amber"], params["confidence"]),
                use_container_width=True)
        with c2:
            st.plotly_chart(
                chart_distribution(res["raw"]["vp_tot"], "Volume-Price Risk",
                                   COL["rose"], params["confidence"]),
                use_container_width=True)

        total_loss = res["raw"]["imb_tot"] + res["raw"]["vp_tot"]
        st.plotly_chart(
            chart_distribution(total_loss, "Combined Portfolio Risk",
                               COL["blue"], params["confidence"]),
            use_container_width=True)

        c3, c4 = st.columns(2)
        with c3:
            st.plotly_chart(
                chart_convergence(total_loss, params["confidence"]),
                use_container_width=True)
        with c4:
            # Distribution statistics table
            m = risk_metrics(total_loss, params["confidence"])
            stats_df = pd.DataFrame({
                "Statistic": [
                    "Expected Loss  E[L]", "Standard Deviation  σ",
                    f"Value-at-Risk  VaR{int(params['confidence']*100)}",
                    f"Cond. VaR  CVaR{int(params['confidence']*100)}",
                    "Unexpected Loss  UL",
                    "Skewness", "Excess Kurtosis",
                    "Paths N",
                ],
                "Value": [
                    f"{m['EL']:,.0f} €",   f"{m['std']:,.0f} €",
                    f"{m['VaR']:,.0f} €",  f"{m['CVaR']:,.0f} €",
                    f"{m['UL']:,.0f} €",
                    f"{m['skew']:.3f}",    f"{m['kurtosis']:.3f}",
                    f"{params['paths']:,}",
                ],
            })
            st.markdown("##### Distribution Diagnostics")
            st.dataframe(stats_df, hide_index=True, use_container_width=True)

    # ╌╌╌╌╌╌╌╌╌╌  TAB 3 — DATA  ╌╌╌╌╌╌╌╌╌╌
    with tab3:
        st.plotly_chart(chart_profile(st.session_state.data, 720),
                        use_container_width=True)

        # Monthly averages
        d = st.session_state.data
        pdf = pd.DataFrame({"date": d["dates"], "load": d["load"], "hpfc": d["hpfc"]})
        pdf["month"] = pdf["date"].apply(lambda x: x.strftime("%Y-%m"))
        monthly = pdf.groupby("month").agg(
            avg_load=("load", "mean"),
            sum_load=("load", "sum"),
            avg_hpfc=("hpfc", "mean"),
        ).reset_index()

        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Bar(
            x=monthly["month"], y=monthly["sum_load"], name="Volume (MWh)",
            marker_color=COL["cyan"], opacity=0.7), secondary_y=False)
        fig.add_trace(go.Scatter(
            x=monthly["month"], y=monthly["avg_hpfc"], name="Avg HPFC (€/MWh)",
            line=dict(color=COL["amber"], width=2.5), mode="lines+markers",
            marker=dict(size=5)), secondary_y=True)
        fig.update_layout(**_layout(
            title=dict(text="Monthly Volume & Average HPFC", font=dict(size=14)),
            height=380,
            legend=dict(orientation="h", y=1.10, x=.5, xanchor="center"),
        ))
        fig.update_yaxes(title_text="MWh", secondary_y=False)
        fig.update_yaxes(title_text="€/MWh", secondary_y=True)
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("Raw data preview  (first 200 rows)"):
            st.dataframe(pdf.head(200), use_container_width=True, height=300)

    # ╌╌╌╌╌╌╌╌╌╌  TAB 4 — METHODOLOGY  ╌╌╌╌╌╌╌╌╌╌
    with tab4:
        st.markdown(METHODOLOGY_MD, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("##### Current Parameterisation")
        param_df = pd.DataFrame([
            ("κ  (mean reversion)",  f"{params['kappa']:.3f}  h⁻¹"),
            ("σ_S  (price diffusion)", f"{params['sigma_price']:.1f}  €/MWh"),
            ("λ  (jump intensity)",  f"{params['jump_prob']:.3f}  h⁻¹"),
            ("σ_J  (jump size)",     f"{params['jump_size']:.0f}  €/MWh"),
            ("φ  (AR persistence)",  f"{params['phi']:.3f}"),
            ("σ_V  (volume error)",  f"{params['vol_error']:.1f} %"),
            ("ρ  (correlation)",     f"{params['correlation']:.0f} %"),
            ("α  (confidence)",      f"{params['confidence']:.1%}"),
            ("r_EC  (cost of equity)", f"{params['cost_of_capital']:.1f} %"),
            ("N  (MC paths)",        f"{params['paths']:,}"),
        ], columns=["Parameter", "Value"])
        st.dataframe(param_df, hide_index=True, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    main()
