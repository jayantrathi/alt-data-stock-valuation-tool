import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from data_pipeline import build_signal_matrix

CORRELATION_THRESHOLD = 0.15
SIGNIFICANCE_THRESHOLD = 0.05
LAG_DAYS = [1, 3, 5, 10, 21, 42, 63]

def analyse_signal_correlations(df, ticker):

    print(f"\n{'='*58}")
    print(f"  Signal Correlation Analysis — {ticker}")
    print(f"{'='*58}")

    exclude_cols = [
        "price", "volume",
        "return_1w", "return_2w", "return_1m", "return_3m"
    ]
    signal_cols = [c for c in df.columns if c not in exclude_cols]
    target_col = "return_1m"

    print(f"  Signals to analyse: {len(signal_cols)}")
    print(f"  Target: {target_col}")
    print(f"  Lags tested: {LAG_DAYS} days")
    print(f"  Data points: {len(df)}")

    results = []

    for signal in signal_cols:
        signal_data = df[signal].dropna()
        target_data = df[target_col].dropna()

        aligned = pd.concat(
            [signal_data, target_data], axis=1
        ).dropna()

        if len(aligned) < 60:
            continue

        signal_vals = aligned.iloc[:, 0]
        target_vals = aligned.iloc[:, 1]

        best_lag = 0
        best_corr = 0
        best_pval = 1.0
        lag_results = {}

        for lag in LAG_DAYS:

            if lag == 0:
                lagged_signal = signal_vals
                lagged_target = target_vals
            else:

                lagged_signal = signal_vals.iloc[:-lag]
                lagged_target = target_vals.iloc[lag:]

                min_len = min(len(lagged_signal), len(lagged_target))
                lagged_signal = lagged_signal.iloc[:min_len]
                lagged_target = lagged_target.iloc[:min_len]

            if len(lagged_signal) < 30:
                continue

            try:
                corr, pval = stats.pearsonr(lagged_signal, lagged_target)
                lag_results[lag] = {"corr": corr, "pval": pval}

                if abs(corr) > abs(best_corr):
                    best_corr = corr
                    best_pval = pval
                    best_lag = lag
            except:
                continue

        if not lag_results:
            continue

        is_significant = best_pval < SIGNIFICANCE_THRESHOLD
        is_meaningful = abs(best_corr) > CORRELATION_THRESHOLD
        direction = "POSITIVE" if best_corr > 0 else "NEGATIVE"

        if abs(best_corr) > 0.4 and is_significant:
            strength = "STRONG"
        elif abs(best_corr) > 0.25 and is_significant:
            strength = "MODERATE"
        elif abs(best_corr) > 0.15 and is_significant:
            strength = "WEAK"
        else:
            strength = "NOISE"

        results.append({
            "signal": signal,
            "best_corr": best_corr,
            "best_lag": best_lag,
            "best_pval": best_pval,
            "is_significant": is_significant,
            "is_meaningful": is_meaningful,
            "direction": direction,
            "strength": strength,
            "lag_results": lag_results,
            "abs_corr": abs(best_corr)
        })

    results = sorted(results, key=lambda x: x["abs_corr"], reverse=True)

    return results

def print_signal_report(results, ticker):
    print(f"\n{'='*58}")
    print(f"  SIGNAL DISCOVERY REPORT — {ticker}")
    print(f"{'='*58}")

    strong = [r for r in results if r["strength"] == "STRONG"]
    moderate = [r for r in results if r["strength"] == "MODERATE"]
    weak = [r for r in results if r["strength"] == "WEAK"]
    noise = [r for r in results if r["strength"] == "NOISE"]

    print(f"\n  Summary:")
    print(f"  Strong signals:   {len(strong)}")
    print(f"  Moderate signals: {len(moderate)}")
    print(f"  Weak signals:     {len(weak)}")
    print(f"  Noise:            {len(noise)}")

    print(f"\n  {'─'*54}")
    print(f"  {'SIGNAL':<28} {'CORR':>7} {'LAG':>6} {'P-VAL':>8} {'STRENGTH'}")
    print(f"  {'─'*54}")

    for r in results:
        if r["strength"] == "NOISE":
            continue
        lag_str = f"{r['best_lag']}d"
        pval_str = f"{r['best_pval']:.4f}"
        sig_marker = "✓" if r["is_significant"] else "✗"
        print(
            f"  {r['signal']:<28} "
            f"{r['best_corr']:>+7.3f} "
            f"{lag_str:>6} "
            f"{pval_str:>8} "
            f"{sig_marker} {r['strength']}"
        )

    print(f"\n  {'─'*54}")
    print(f"  ✓ = statistically significant (p < {SIGNIFICANCE_THRESHOLD})")
    print(f"  Lag = how many days signal leads the return")

    if strong:
        print(f"\n  TOP PREDICTIVE SIGNALS:")
        for r in strong[:5]:
            direction_emoji = "↑" if r["direction"] == "POSITIVE" else "↓"
            print(f"\n  {direction_emoji} {r['signal']}")
            print(f"     Correlation: {r['best_corr']:+.3f} at {r['best_lag']}-day lag")
            print(f"     P-value:     {r['best_pval']:.4f}")
            print(f"     Meaning:     When this signal rises, "
                  f"{ticker} tends to {'rise' if r['direction'] == 'POSITIVE' else 'fall'} "
                  f"{r['best_lag']} days later")

    return results

def analyse_lag_structure(results, ticker):

    print(f"\n{'='*58}")
    print(f"  LAG STRUCTURE — Top 5 Signals")
    print(f"{'='*58}")

    top_signals = [r for r in results
                   if r["strength"] in ["STRONG", "MODERATE"]][:5]

    if not top_signals:
        print("  No strong/moderate signals found.")
        return

    for r in top_signals:
        print(f"\n  {r['signal']}:")
        print(f"  {'Lag':<8} {'Correlation':>12} {'P-value':>10} {'Signal'}")
        for lag, data in sorted(r["lag_results"].items()):
            bar_len = int(abs(data["corr"]) * 20)
            bar = "█" * bar_len
            sig = "✓" if data["pval"] < SIGNIFICANCE_THRESHOLD else " "
            direction = "+" if data["corr"] > 0 else "-"
            print(f"  {lag}d{'':<5} {data['corr']:>+12.3f} "
                  f"{data['pval']:>10.4f} {sig} {direction}{bar}")

def rolling_correlation_analysis(df, top_signals, ticker):

    print(f"\n{'='*58}")
    print(f"  ROLLING CORRELATION STABILITY")
    print(f"{'='*58}")

    target = df["return_1m"]
    window = 126

    stability_results = []

    for signal_name in top_signals[:5]:
        if signal_name not in df.columns:
            continue

        signal = df[signal_name]
        aligned = pd.concat([signal, target], axis=1).dropna()

        if len(aligned) < window * 2:
            print(f"  {signal_name}: insufficient data for rolling analysis")
            continue

        rolling_corr = aligned.iloc[:, 0].rolling(window).corr(
            aligned.iloc[:, 1]
        )

        mean_corr = rolling_corr.dropna().mean()
        std_corr = rolling_corr.dropna().std()

        overall_corr = aligned.iloc[:, 0].corr(aligned.iloc[:, 1])
        consistency = (
            (rolling_corr.dropna() > 0).mean()
            if overall_corr > 0
            else (rolling_corr.dropna() < 0).mean()
        )

        if std_corr < 0.15:
            stability = "STABLE"
        elif std_corr < 0.25:
            stability = "MODERATE"
        else:
            stability = "UNSTABLE"

        print(f"\n  {signal_name}:")
        print(f"    Mean rolling corr:  {mean_corr:+.3f}")
        print(f"    Std of rolling corr: {std_corr:.3f}")
        print(f"    Directional consistency: {consistency*100:.0f}%")
        print(f"    Stability: {stability}")

        stability_results.append({
            "signal": signal_name,
            "mean_corr": mean_corr,
            "std_corr": std_corr,
            "consistency": consistency,
            "stability": stability,
            "rolling_corr": rolling_corr
        })

    return stability_results

def build_composite_score(df, results, ticker):

    print(f"\n{'='*58}")
    print(f"  BUILDING COMPOSITE SIGNAL SCORE")
    print(f"{'='*58}")

    sig_results = [
        r for r in results
        if r["is_significant"] and r["strength"] != "NOISE"
    ][:8]

    if not sig_results:
        print("  No significant signals to combine.")
        return None

    print(f"  Combining {len(sig_results)} significant signals...")

    composite = pd.Series(0.0, index=df.index)
    total_weight = 0

    for r in sig_results:
        signal_name = r["signal"]
        if signal_name not in df.columns:
            continue

        signal = df[signal_name].copy()

        signal_mean = signal.mean()
        signal_std = signal.std()
        if signal_std == 0:
            continue
        signal_z = (signal - signal_mean) / signal_std

        if r["best_corr"] < 0:
            signal_z = -signal_z

        weight = r["abs_corr"]
        composite += signal_z * weight
        total_weight += weight

        print(f"  + {signal_name:<28} weight: {weight:.3f}")

    if total_weight == 0:
        return None

    composite = composite / total_weight
    composite_scaled = composite.rank(pct=True) * 200 - 100

    latest_score = composite_scaled.dropna().iloc[-1]
    print(f"\n  Current composite score: {latest_score:+.1f} / 100")

    if latest_score > 50:
        signal_label = "STRONGLY BULLISH"
    elif latest_score > 20:
        signal_label = "BULLISH"
    elif latest_score > -20:
        signal_label = "NEUTRAL"
    elif latest_score > -50:
        signal_label = "BEARISH"
    else:
        signal_label = "STRONGLY BEARISH"

    print(f"  Signal interpretation: {signal_label}")

    return composite_scaled

def visualise_signals(df, results, composite_score,
                      stability_results, ticker):
    print(f"\n  Generating visualisations...")

    fig = plt.figure(figsize=(20, 16))
    fig.suptitle(
        f"{ticker} — Signal Discovery Engine",
        fontsize=16, fontweight="bold", y=0.98
    )
    gs = gridspec.GridSpec(3, 3, figure=fig,
                           hspace=0.45, wspace=0.35)

    ax1 = fig.add_subplot(gs[0, :2])
    ax2 = fig.add_subplot(gs[0, 2])
    ax3 = fig.add_subplot(gs[1, :2])
    ax4 = fig.add_subplot(gs[1, 2])
    ax5 = fig.add_subplot(gs[2, :])

    top_results = [r for r in results
                   if r["strength"] != "NOISE"][:15]
    if top_results:
        signals = [r["signal"][:20] for r in top_results]
        corrs = [r["best_corr"] for r in top_results]
        colors = ["#27AE60" if c > 0 else "#E74C3C" for c in corrs]
        alpha = [1.0 if r["is_significant"] else 0.4
                 for r in top_results]

        bars = ax1.barh(signals, corrs, color=colors, alpha=0.8)
        ax1.axvline(0, color="black", linewidth=0.8)
        ax1.axvline(CORRELATION_THRESHOLD, color="gray",
                    linewidth=1, linestyle="--", alpha=0.5)
        ax1.axvline(-CORRELATION_THRESHOLD, color="gray",
                    linewidth=1, linestyle="--", alpha=0.5)
        ax1.set_xlabel("Correlation with 1-month forward return")
        ax1.set_title("Signal Correlations (solid = significant)")
        ax1.grid(True, alpha=0.2, axis="x")

    strength_counts = {
        "Strong": len([r for r in results
                       if r["strength"] == "STRONG"]),
        "Moderate": len([r for r in results
                         if r["strength"] == "MODERATE"]),
        "Weak": len([r for r in results
                     if r["strength"] == "WEAK"]),
        "Noise": len([r for r in results
                      if r["strength"] == "NOISE"]),
    }
    colors_pie = ["#27AE60", "#F39C12", "#3498DB", "#95A5A6"]
    wedges = [v for v in strength_counts.values() if v > 0]
    labels = [k for k, v in strength_counts.items() if v > 0]
    if wedges:
        ax2.pie(wedges, labels=labels, colors=colors_pie[:len(wedges)],
                autopct="%1.0f%%", startangle=90)
        ax2.set_title("Signal Quality Distribution")

    if composite_score is not None:
        price = df["price"]
        ax3_twin = ax3.twinx()

        price_norm = (price - price.mean()) / price.std()
        composite_norm = composite_score / 100

        ax3.plot(composite_score.index, composite_score,
                 color="#4A90D9", linewidth=1.5, alpha=0.8,
                 label="Composite signal score")
        ax3.axhline(0, color="gray", linewidth=0.8, linestyle="--")
        ax3.fill_between(composite_score.index, composite_score, 0,
                         where=composite_score > 0,
                         alpha=0.15, color="#27AE60")
        ax3.fill_between(composite_score.index, composite_score, 0,
                         where=composite_score < 0,
                         alpha=0.15, color="#E74C3C")
        ax3_twin.plot(price.index, price,
                      color="#E74C3C", linewidth=1.5,
                      alpha=0.6, label="Stock price")
        ax3.set_ylabel("Composite score (-100 to +100)")
        ax3_twin.set_ylabel("Stock price ($)")
        ax3.set_title("Composite Signal Score vs Stock Price")

        lines1, labels1 = ax3.get_legend_handles_labels()
        lines2, labels2 = ax3_twin.get_legend_handles_labels()
        ax3.legend(lines1 + lines2, labels1 + labels2,
                   loc="upper left", fontsize=9)

    top5 = [r for r in results
            if r["strength"] in ["STRONG", "MODERATE"]][:5]
    if top5:
        lag_matrix = []
        row_labels = []
        for r in top5:
            row = [r["lag_results"].get(lag, {}).get("corr", 0)
                   for lag in LAG_DAYS]
            lag_matrix.append(row)
            row_labels.append(r["signal"][:18])

        lag_df = pd.DataFrame(
            lag_matrix,
            index=row_labels,
            columns=[f"{l}d" for l in LAG_DAYS]
        )
        sns.heatmap(
            lag_df, ax=ax4, cmap="RdYlGn", center=0,
            annot=True, fmt=".2f", linewidths=0.5,
            cbar_kws={"shrink": 0.8}
        )
        ax4.set_title("Correlation by Lag (days)")
        ax4.set_xlabel("Lag")

    if stability_results:
        most_stable = stability_results[0]
        rolling_corr = most_stable["rolling_corr"]
        ax5.plot(rolling_corr.index, rolling_corr,
                 color="#4A90D9", linewidth=1.5)
        ax5.axhline(0, color="black", linewidth=0.8)
        ax5.axhline(CORRELATION_THRESHOLD, color="#27AE60",
                    linewidth=1, linestyle="--", alpha=0.6,
                    label=f"Threshold ({CORRELATION_THRESHOLD})")
        ax5.axhline(-CORRELATION_THRESHOLD, color="#E74C3C",
                    linewidth=1, linestyle="--", alpha=0.6)
        ax5.fill_between(rolling_corr.index, rolling_corr, 0,
                         where=rolling_corr > 0,
                         alpha=0.15, color="#27AE60")
        ax5.fill_between(rolling_corr.index, rolling_corr, 0,
                         where=rolling_corr < 0,
                         alpha=0.15, color="#E74C3C")
        ax5.set_title(
            f"Rolling 6-Month Correlation: "
            f"{most_stable['signal']} vs Forward Returns"
        )
        ax5.set_ylabel("Correlation")
        ax5.legend(fontsize=9)
        ax5.grid(True, alpha=0.2)

    save = input("\nSave chart? (y/n): ").strip().lower()
    if save == "y":
        plt.savefig(f"{ticker}_signals.png", dpi=150,
                    bbox_inches="tight")
        print(f"  Saved as {ticker}_signals.png")
    plt.show()

def main():
    ticker = input("Enter ticker: ").upper().strip()

    import os
    csv_file = f"{ticker}_signals.csv"
    if os.path.exists(csv_file):
        print(f"\n  Loading existing signal matrix from {csv_file}...")
        df = pd.read_csv(csv_file, index_col=0, parse_dates=True)
        company_name = ticker
    else:
        print(f"\n  No existing data found — running pipeline...")
        df, company_name = build_signal_matrix(ticker)

    print(f"\n  Signal matrix: {df.shape[0]} rows × {df.shape[1]} cols")

    results = analyse_signal_correlations(df, ticker)
    results = print_signal_report(results, ticker)
    analyse_lag_structure(results, ticker)

    top_signal_names = [
        r["signal"] for r in results
        if r["strength"] in ["STRONG", "MODERATE"]
    ][:5]
    stability_results = rolling_correlation_analysis(
        df, top_signal_names, ticker
    )

    composite_score = build_composite_score(df, results, ticker)

    visualise_signals(
        df, results, composite_score, stability_results, ticker
    )

if __name__ == "__main__":
    main()
