import matplotlib
matplotlib.use("Agg")
import matplotlib; matplotlib.use('Agg')
"""
vol_surface.py
--------------
3D implied volatility surface + vol smile charts for SOXL puts.
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from scipy.interpolate import griddata
from pathlib import Path


def plot_price_history(data, save=True):
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(data.index, data["Close"], color="#1D9E75", linewidth=1.8)
    ax.fill_between(data.index, data["Close"], alpha=0.08, color="#1D9E75")
    ax.set_title("SOXL Daily Close Price (1Y)", fontsize=13, fontweight='bold')
    ax.set_xlabel("Date")
    ax.set_ylabel("Price ($)")
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    if save:
        Path("data").mkdir(exist_ok=True)
        plt.savefig("data/price_history.png", dpi=150)
        print("Saved: data/price_history.png")
    #plt.show()


def plot_vol_surface_3d(df, spot_price, save=True):
    x = df['moneyness'].values
    y = df['DTE'].values
    z = df['impliedVolatility'].values

    xi = np.linspace(x.min(), x.max(), 60)
    yi = np.linspace(y.min(), y.max(), 60)
    xi, yi = np.meshgrid(xi, yi)
    zi = griddata((x, y), z, (xi, yi), method='linear')

    fig = plt.figure(figsize=(13, 8))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(xi, yi, zi, cmap='RdYlGn_r', alpha=0.85, edgecolor='none')

    ax.set_xlabel('Moneyness (K/S)', fontsize=10, labelpad=8)
    ax.set_ylabel('Days to Expiration', fontsize=10, labelpad=8)
    ax.set_zlabel('Implied Volatility', fontsize=10, labelpad=8)
    ax.set_title(f'SOXL Implied Volatility Surface\nSpot: ${spot_price:.2f}',
                 fontsize=13, fontweight='bold')
    fig.colorbar(surf, shrink=0.4, label='IV')

    plt.tight_layout()
    if save:
        Path("data").mkdir(exist_ok=True)
        plt.savefig("data/vol_surface_3d.png", dpi=150, bbox_inches='tight')
        print("Saved: data/vol_surface_3d.png")
    #plt.show()


def plot_vol_surface_interactive(df, spot_price, save=True):
    x = df['moneyness'].values
    y = df['DTE'].values
    z = df['impliedVolatility'].values

    xi = np.linspace(x.min(), x.max(), 60)
    yi = np.linspace(y.min(), y.max(), 60)
    xi_grid, yi_grid = np.meshgrid(xi, yi)
    zi_grid = griddata((x, y), z, (xi_grid, yi_grid), method='linear')

    fig = go.Figure(data=[go.Surface(
        x=xi, y=yi, z=zi_grid,
        colorscale='RdYlGn', reversescale=True,
        colorbar=dict(title='IV'),
        hovertemplate='Moneyness: %{x:.2f}<br>DTE: %{y}<br>IV: %{z:.3f}<extra></extra>'
    )])

    fig.update_layout(
        title=dict(text=f'SOXL Implied Volatility Surface — Spot ${spot_price:.2f}',
                   font=dict(size=14)),
        scene=dict(
            xaxis_title='Moneyness (K/S)',
            yaxis_title='Days to Expiration',
            zaxis_title='Implied Volatility',
            camera=dict(eye=dict(x=1.5, y=-1.5, z=0.8))
        ),
        width=900, height=650,
    )

    if save:
        Path("data").mkdir(exist_ok=True)
        fig.write_html("data/vol_surface_interactive.html")
        print("Saved: data/vol_surface_interactive.html")
    #fig.show()


def plot_vol_smile(df, spot_price, save=True):
    expirations = sorted(df['expiration'].unique())[:6]
    n = len(expirations)
    cols = 3
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    axes = axes.flatten()

    for i, exp in enumerate(expirations):
        exp_data = df[df['expiration'] == exp].sort_values('moneyness')
        dte = int(exp_data['DTE'].iloc[0])

        axes[i].scatter(exp_data['moneyness'], exp_data['impliedVolatility'],
                        color='#1D9E75', s=20, alpha=0.7)
        axes[i].plot(exp_data['moneyness'], exp_data['impliedVolatility'],
                     color='#1D9E75', linewidth=1.5, alpha=0.6)
        axes[i].axvline(x=1.0, color='red', linestyle='--',
                        alpha=0.5, linewidth=1, label='ATM')
        axes[i].set_title(f'{exp}  ({dte} DTE)', fontsize=11)
        axes[i].set_xlabel('Moneyness')
        axes[i].set_ylabel('Implied Volatility')
        axes[i].legend(fontsize=8)
        axes[i].grid(True, alpha=0.3)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle('SOXL Volatility Smile by Expiration',
                 fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    if save:
        Path("data").mkdir(exist_ok=True)
        plt.savefig("data/vol_smile.png", dpi=150, bbox_inches='tight')
        print("Saved: data/vol_smile.png")
    #plt.show()