import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from statsmodels.nonparametric.smoothers_lowess import lowess
from scipy.interpolate import interp1d
from matplotlib.ticker import StrMethodFormatter

# -----------------------------
# Utility Functions
# -----------------------------
def rgompertz(n, a, b):
    u = np.random.uniform(0, 1, n)
    return (1 / a) * np.log(1 - (a / b) * np.log(1 - u))

def calculate_pv(lifespan_months, premium_monthly, face_value, cash_value, discount_rate_monthly):
    pv_list = []
    for L in lifespan_months:
        L = int(np.floor(L))
        remaining_cash = cash_value
        cash_flow = 0
        for _ in range(L):
            if remaining_cash > 0:
                if remaining_cash >= premium_monthly:
                    remaining_cash -= premium_monthly
                else:
                    cash_flow -= (premium_monthly - remaining_cash)
                    remaining_cash = 0
            else:
                cash_flow -= premium_monthly
            cash_flow /= (1 + discount_rate_monthly)
        cash_flow += face_value / ((1 + discount_rate_monthly) ** L)
        pv_list.append(cash_flow)
    return np.array(pv_list)

def months_since(date_str):
    le_date = datetime.strptime(date_str, "%Y-%m-%d")
    today = datetime.today()
    return max(0, (today.year - le_date.year) * 12 + today.month - le_date.month)

def run_simulation(name, lifespans, premium_monthly, face_value, cash_value, discount_rate_monthly, le_adjusted):
    pv = calculate_pv(lifespans, premium_monthly, face_value, cash_value, discount_rate_monthly)
    df = pd.DataFrame({'Lifespan': lifespans, 'PresentValue': pv})
    
    # Percentiles
    ls_25, ls_50, ls_75 = np.percentile(df['Lifespan'], [25, 50, 75])
    pv_25, pv_50, pv_75 = np.percentile(df['PresentValue'], [25, 50, 75])
    
    # Interpolation
    smooth = lowess(df['PresentValue'], df['Lifespan'], frac=0.1, return_sorted=True)
    lifespan_smooth = smooth[:, 0]
    value_smooth = smooth[:, 1]
    interp_func = interp1d(lifespan_smooth, value_smooth, bounds_error=False, fill_value="extrapolate")
    interpolated_val = float(interp_func(le_adjusted))

    stats = {
        'lifespan': [ls_25, ls_50, ls_75],
        'pv': [pv_25, pv_50, pv_75],
        'interpolated': interpolated_val,
        'df': df,
        'curve': (lifespan_smooth, value_smooth),
        'label': name
    }
    return stats

def plot_side_by_side(gom_stats, wei_stats):
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))

    for i, stats in enumerate([gom_stats, wei_stats]):
        ax = axes[i]
        df = stats['df']
        ax.scatter(df['Lifespan'], df['PresentValue'], alpha=0.4, color='blue', label="Simulations")
        ax.plot(*stats['curve'], color='red', label="LOESS Trend")

        colors = ['green', 'orange', 'purple']
        for idx, ls in enumerate(stats['lifespan']):
            y_val = float(interp1d(*stats['curve'])(ls))
            ax.plot([ls, ls], [0, y_val], linestyle='dotted', color=colors[idx])
            ax.annotate(f"LE: {round(ls, 1)} mo", xy=(ls + 2, y_val * 0.03), color=colors[idx])

        for idx, pv in enumerate(stats['pv']):
            x_val = float(interp1d(stats['curve'][1], stats['curve'][0])(pv))
            ax.plot([0, x_val], [pv, pv], linestyle='dashed', color=colors[2 - idx])
            ax.annotate(f"Val: ${round(pv):,.0f}", xy=(x_val + 2, pv + 3000), color=colors[2 - idx])

        ax.set_xlim(0, 200)
        ax.set_title(stats['label'])
        ax.set_xlabel("Life Expectancy (Months)")
        ax.set_ylabel("Policy Valuation at LE Generation ($)")
        ax.grid(True)
        ax.legend()
        ax.yaxis.set_major_formatter(StrMethodFormatter('${x:,.0f}'))

    plt.tight_layout()
    fig.savefig("monte_carlo_simulations.png", dpi=300)
    return fig

# -----------------------
# Streamlit UI Entry Point
# -----------------------
def run_simulation_ui(face_value, cash_value, premium_annual,
                      shape_g, scale_g_annual, shape_w, scale_w_annual,
                      le_at_report, le_report_date):

    premium_monthly = premium_annual / 12
    discount_rate_monthly = 0.05 / 12
    months_elapsed = months_since(le_report_date)
    le_adjusted = le_at_report - months_elapsed

    np.random.seed(42)
    n_sim = 25000

    # ✅ Gompertz scale must be divided by 12 (annual → monthly)
    lifespans_g = rgompertz(n_sim, shape_g, scale_g_annual / 12) * 12
    # ✅ Weibull scale must be multiplied by 12 (annual → monthly)
    lifespans_w = np.random.weibull(shape_w, n_sim) * (scale_w_annual * 12)

    gom_stats = run_simulation("Valuation with fitted Gompertz Mortality Curve", lifespans_g, premium_monthly, face_value, cash_value, discount_rate_monthly, le_adjusted)
    wei_stats = run_simulation("Valuation with fitted Weibull Mortality Curve", lifespans_w, premium_monthly, face_value, cash_value, discount_rate_monthly, le_adjusted)

    fig = plot_side_by_side(gom_stats, wei_stats)
    return gom_stats, wei_stats, fig, le_adjusted
