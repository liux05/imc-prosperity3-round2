# Basket Statistical Arbitrage System - Complete Guide

## 📋 Table of Contents
1. [Quick Start](#quick-start)
2. [System Architecture](#system-architecture)
3. [Core Concepts](#core-concepts)
4. [Usage Examples](#usage-examples)
5. [Parameter Tuning](#parameter-tuning)
6. [Interpretation Guide](#interpretation-guide)

---

## 🚀 Quick Start

### Installation
```bash
pip install numpy pandas scipy statsmodels matplotlib seaborn
pip install ruptures hmmlearn  # Optional for advanced features
```

### Basic Usage
```python
import pandas as pd
from basket_stat_arb import (
    build_baskets_and_spread,
    backtest,
    BacktestConfig,
    print_backtest_summary
)

# Load your data
prices = pd.read_csv('your_data.csv')  # Must have columns: C, J, D

# Configure backtest
config = BacktestConfig(
    z_enter=2.0,      # Entry threshold
    z_exit=0.5,       # Exit threshold  
    halflife_mu=50,   # EWMA halflife for mean
    halflife_sigma=50 # EWMA halflife for std
)

# Run backtest
results = backtest(prices, config)
print_backtest_summary(results)

# Visualize
from basket_arb_viz import plot_backtest_results, plot_component_analysis
plot_backtest_results(results)
plot_component_analysis(results)
```

---

## 🏗️ System Architecture

### Module 1: Core Engine (`basket_stat_arb.py`)

**Basket Construction**
- `build_baskets_and_spread()`: Creates B1, B2, and mispricing spread S

**Mean Reversion Testing**
- `adf_test()`: Augmented Dickey-Fuller stationarity test
- `hurst_exponent()`: R/S analysis for mean reversion
- `variance_ratio_test()`: Lo-MacKinlay test
- `calculate_half_life()`: Speed of mean reversion

**Regime Detection**
- `rolling_corr_and_fisher()`: Fisher z-transform with confidence intervals
- `classify_regime_rules()`: Rules-based regime classifier
- `distance_metrics()`: Euclidean distance between normalized series

**Signal Generation**
- `compute_zscore_S()`: EWMA-based z-score calculation
- `generate_signals()`: Entry/exit logic with regime gating
- `map_positions_to_legs()`: Convert arb units to actual positions

**Backtesting**
- `backtest()`: Complete simulation with costs and risk controls

### Module 2: Visualization (`basket_arb_viz.py`)

- `plot_backtest_results()`: Comprehensive 8-panel dashboard
- `plot_component_analysis()`: Component relationship visualization
- `parameter_sensitivity_analysis()`: Single-parameter sweep
- `run_parameter_sweep()`: Multi-parameter grid search
- `detect_change_points()`: PELT/BinSeg change-point detection
- `classify_regime_hmm()`: HMM-based regime detection

---

## 💡 Core Concepts

### The Deterministic Replication Identity

**Key Insight**: The baskets have a **deterministic relationship** (not statistical):

```
B1 = 6×C + 3×J + 1×D
B2 = 4×C + 2×J

Mathematical identity:
B1 = 1.5 × B2 + D
```

**Mispricing Spread**:
```
S_t = B1_t - (1.5×B2_t + D_t)
```

In a frictionless market, `S ≡ 0`. Any deviation is a **pricing error** (not model error).

This is **stronger than cointegration** because:
- No parameter uncertainty
- No estimation error
- Exact replication (not statistical relationship)

### Mean Reversion Tests

**1. ADF Test (Stationarity)**
- H0: Unit root (non-stationary, random walk)
- H1: Stationary (mean-reverting)
- **Trade if**: p-value < 0.10 (reject H0)

**2. Hurst Exponent (Behavior)**
- H < 0.5 → Mean-reverting
- H ≈ 0.5 → Random walk
- H > 0.5 → Trending/momentum
- **Trade if**: H < 0.5 (preferably < 0.4)

**3. Variance Ratio (Structure)**
- VR < 1 → Mean reversion
- VR > 1 → Momentum
- **Trade if**: VR < 1 and p-value < 0.10

**4. Half-Life (Speed)**
- Measures how fast spread reverts to mean
- Half-life = -ln(2) / ln(ρ) where ρ is AR(1) coefficient
- **Trade if**: Half-life < 100 bars (must be fast vs holding period and costs)

### Z-Score Interpretation

**Typical Ranges**:
- ±0 to ±2: Normal range
- ±2 to ±4: Entry territory (statistically significant)
- ±4 to ±6: Extreme (rare events)
- ±20+: Likely scaling error or data issue

**Why EWMA**:
- Adaptive to changing market conditions
- More weight on recent data
- Better for non-stationary environments

### Regime Detection

**Correlated Regime (State 0)**:
- Negative correlation between C and J
- Fisher CI away from zero
- Small residual volatility
- Low Euclidean distance
- **Strategy**: Normal trading with standard thresholds

**Divergence Regime (State 1)**:
- Correlation breaks down (CI crosses 0)
- High residual volatility
- Large Euclidean distance
- Recent change points
- **Strategy**: Widen thresholds or flatten positions

**Why It Matters**:
- You may observe visual divergences in your data (e.g., the inverse relationship temporarily breaks down)
- The system detects these algorithmically using statistical criteria
- During these periods, the C-J inverse relationship is unreliable
- Trading the spread during divergence is riskier (component hedge is unstable)
- Regime gating prevents losses during unstable periods

---

## 📚 Usage Examples

### Example 1: Load Your CSV Data

```python
import pandas as pd

# Your CSV should have these columns: timestamp, product, mid_price
# We need to reshape it to wide format with C, J, D columns

# Method 1: If already in wide format
prices = pd.read_csv('apexoa.csv')
prices = prices[['C', 'J', 'D']]  # Select relevant columns

# Method 2: If in long format (timestamp, product, mid_price)
df_long = pd.read_csv('apexoa.csv')
prices = df_long.pivot_table(
    values='mid_price',
    index='timestamp',
    columns='product'
)[['CROISSANTS', 'JAMS', 'DJEMBES']]
prices.columns = ['C', 'J', 'D']

print(f"Loaded {len(prices)} rows")
print(prices.head())
```

### Example 2: Analyze Mean Reversion

```python
from basket_stat_arb import (
    build_baskets_and_spread,
    mean_reversion_tests,
    fit_pair_model_rolling_ols
)

# Build baskets
df = build_baskets_and_spread(prices)

# Test spread S for mean reversion
mr_stats = mean_reversion_tests(df['S'])
print(mr_stats)

# Test component residual
residuals, betas = fit_pair_model_rolling_ols(df['C'], df['J'], window=50)
mr_stats_comp = mean_reversion_tests(residuals)
print(mr_stats_comp)

# Interpretation
if mr_stats.is_tradeable:
    print("✓ Spread S is tradeable!")
    print(f"  - ADF p-value: {mr_stats.adf_pvalue:.4f} (< 0.10 ✓)")
    print(f"  - Hurst: {mr_stats.hurst_exponent:.3f} (< 0.5 ✓)")
    print(f"  - Half-life: {mr_stats.half_life:.1f} bars")
else:
    print("✗ Spread may not be tradeable")
```

### Example 3: Full Backtest with Visualization

```python
from basket_stat_arb import backtest, BacktestConfig, print_backtest_summary
from basket_arb_viz import plot_backtest_results, plot_component_analysis
import matplotlib.pyplot as plt

# Configure
config = BacktestConfig(
    z_enter=2.0,
    z_exit=0.5,
    z_cap=4.0,
    halflife_mu=50,
    halflife_sigma=50,
    regime_method='rules',
    max_gross_exposure=1.0,
    commission_bps=1.0,
    half_spread_bps=2.0
)

# Run backtest
results = backtest(prices, config)

# Print summary
print_backtest_summary(results)

# Visualize
fig1 = plot_backtest_results(results)
fig2 = plot_component_analysis(results)
plt.show()

# Access specific results
print(f"\nFinal P&L: {results['metrics']['total_return']:.2f}")
print(f"Sharpe Ratio: {results['metrics']['sharpe_ratio']:.2f}")
print(f"Max Drawdown: {results['metrics']['max_drawdown']:.2%}")
print(f"Win Rate: {results['metrics']['win_rate']:.2%}")
```

### Example 4: Parameter Sensitivity Analysis

```python
from basket_arb_viz import (
    parameter_sensitivity_analysis,
    plot_parameter_sensitivity
)

# Test different z_enter thresholds
z_values = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
sensitivity_df = parameter_sensitivity_analysis(
    prices,
    config,
    param_name='z_enter',
    param_values=z_values,
    backtest_func=backtest
)

print(sensitivity_df)

# Visualize
fig = plot_parameter_sensitivity(sensitivity_df, 'z_enter')
plt.show()

# Find optimal
best_sharpe_idx = sensitivity_df['sharpe_ratio'].idxmax()
best_config = sensitivity_df.iloc[best_sharpe_idx]
print(f"\nOptimal z_enter: {best_config['z_enter']}")
print(f"Sharpe: {best_config['sharpe_ratio']:.2f}")
```

### Example 5: Multi-Parameter Grid Search

```python
from basket_arb_viz import run_parameter_sweep, plot_parameter_heatmap

# Define parameter grid
param_grid = {
    'z_enter': [1.5, 2.0, 2.5, 3.0],
    'z_exit': [0.3, 0.5, 0.7],
    'halflife_mu': [30, 50, 70]
}

# Run sweep (this may take a while)
sweep_results = run_parameter_sweep(
    prices,
    config,
    backtest_func=backtest,
    param_grid=param_grid
)

# Find best configuration
best_idx = sweep_results['sharpe_ratio'].idxmax()
best_params = sweep_results.iloc[best_idx]
print("Best Parameters:")
print(best_params)

# Visualize 2D heatmap
fig = plot_parameter_heatmap(
    sweep_results,
    param1='z_enter',
    param2='z_exit',
    metric='sharpe_ratio'
)
plt.show()
```

### Example 6: Change Point Detection

```python
from basket_arb_viz import detect_change_points, plot_change_points

# Build baskets
df = build_baskets_and_spread(prices)

# Detect change points in spread
change_points = detect_change_points(
    df['S'],
    method='pelt',
    penalty=10.0
)

print(f"Detected {len(change_points)} change points")
print(f"Change point timestamps: {change_points}")

# Visualize
fig = plot_change_points(
    df['S'],
    change_points,
    title="Change Points in Mispricing Spread"
)
plt.show()

# Check if your observed divergences align
print("\nYour observed divergences were at ~12804 and ~16666")
print("Do detected change points align with these?")
```

### Example 7: HMM Regime Detection

```python
from basket_arb_viz import classify_regime_hmm
from basket_stat_arb import (
    rolling_corr_and_fisher,
    compute_ewma_zscore,
    distance_metrics
)

# Build features for HMM
df = build_baskets_and_spread(prices)

# Correlation features
corr_df = rolling_corr_and_fisher(df['C'], df['J'], window=50)

# Normalized distance
c_z = compute_ewma_zscore(df['C'], halflife=50)
j_z = compute_ewma_zscore(df['J'], halflife=50)
j_inv_z = -j_z
dist = distance_metrics(c_z, j_inv_z, window=50)

# Component residual
residuals, betas = fit_pair_model_rolling_ols(df['C'], df['J'], window=50)
residual_z = compute_ewma_zscore(residuals, halflife=50)

# Build feature matrix
features = pd.DataFrame({
    'fisher_z': 0.5 * np.log((1 + corr_df['corr']) / (1 - corr_df['corr'])),
    'residual_z_abs': residual_z.abs(),
    'distance': dist,
    'spread_z_abs': compute_ewma_zscore(df['S'], halflife=50).abs()
})

# Standardize
features_std = (features - features.mean()) / features.std()

# Fit HMM
regime_hmm, model = classify_regime_hmm(features_std, n_states=2)

# Compare with rules-based
from basket_stat_arb import classify_regime_rules
regime_rules = classify_regime_rules(corr_df, residual_z, dist, hysteresis=5)

# Visualize comparison
fig, axes = plt.subplots(2, 1, figsize=(14, 8))

axes[0].plot(df.index, df['S'], label='Spread S', alpha=0.7)
for i in range(len(regime_rules)-1):
    if regime_rules.iloc[i] == 1:
        axes[0].axvspan(regime_rules.index[i], regime_rules.index[i+1], 
                       alpha=0.2, color='red')
axes[0].set_title('Rules-Based Regime Detection')
axes[0].legend()

axes[1].plot(df.index, df['S'], label='Spread S', alpha=0.7)
for i in range(len(regime_hmm)-1):
    if regime_hmm.iloc[i] == 1:
        axes[1].axvspan(regime_hmm.index[i], regime_hmm.index[i+1], 
                       alpha=0.2, color='red')
axes[1].set_title('HMM-Based Regime Detection')
axes[1].legend()

plt.tight_layout()
plt.show()
```

---

## 🎯 Parameter Tuning

### Entry/Exit Thresholds

**z_enter** (Entry Threshold)
- **Range**: 1.5 to 3.0
- **Lower** (1.5-2.0): More trades, higher turnover, more false signals
- **Higher** (2.5-3.0): Fewer trades, stronger signals, may miss opportunities
- **Recommended**: Start with 2.0, adjust based on hit rate and Sharpe

**z_exit** (Exit Threshold)
- **Range**: 0.3 to 0.7
- **Lower** (0.3-0.4): Exit near mean, realize profits quickly
- **Higher** (0.6-0.7): Wait for fuller reversion, risk reversals
- **Recommended**: 0.5 (half a standard deviation)

**z_cap** (Position Size Cap)
- **Range**: 3.0 to 5.0
- Caps position size when z-score is extreme
- Prevents overleveraging on outliers
- **Recommended**: 4.0

### EWMA Halflifes

**halflife_mu** (Mean Halflife)
- **Range**: 30 to 100 bars
- **Shorter** (30-50): More adaptive, faster reaction
- **Longer** (70-100): More stable, slower adaptation
- **Recommended**: 50 for medium-term mean

**halflife_sigma** (Volatility Halflife)
- **Range**: 20 to 100 bars
- **Shorter**: Faster vol adjustment (good in volatile regimes)
- **Longer**: Smoother vol estimate (good in stable regimes)
- **Recommended**: 50, or slightly shorter than halflife_mu

### Regime Detection

**rolling_window**
- **Range**: 30 to 100 bars
- Window for rolling correlation and statistics
- **Shorter**: More sensitive to regime changes
- **Longer**: More stable regime classification
- **Recommended**: 50

**hysteresis_bars**
- **Range**: 3 to 10 bars
- Bars required to confirm regime switch
- Prevents regime flipping on noise
- **Recommended**: 5

### Risk Controls

**max_gross_exposure**
- Maximum total notional exposure
- Set based on risk tolerance and capital
- **Recommended**: 1.0 to 3.0

**max_drawdown_pct**
- Kill-switch drawdown threshold
- Flattens all positions if breached
- **Recommended**: 0.10 to 0.20 (10-20%)

**commission_bps** and **half_spread_bps**
- Model your actual trading costs
- Higher costs → need wider entry thresholds
- **Typical**: 1-5 bps commission, 2-5 bps spread

---

## 📊 Interpretation Guide

### Reading the Backtest Dashboard

**Panel 1: Mispricing Spread S**
- Shows raw spread over time
- Should oscillate around zero (red dashed line)
- Large persistent deviations → potential data issues
- Check: Is spread mean-reverting visually?

**Panel 2: Z-Score with Regime Shading**
- Red shading = Divergence regime
- Green/orange lines = Entry/exit thresholds
- Check: Do entries occur at extremes? Do regimes align with divergences?

**Panel 3: Target Position**
- Shows arb unit position over time
- Positive = LONG B1, SHORT 1.5×B2, SHORT D
- Negative = opposite
- Check: Are positions reasonable? Not constantly flipping?

**Panel 4: Cumulative P&L**
- Should trend upward if strategy is profitable
- Drawdowns should be manageable
- Compare to peak (red dashed line)
- Check: Is growth consistent or just a few lucky trades?

**Panel 5: Drawdown**
- Shows underwater equity
- Red area = losing periods
- Check: Max drawdown vs your risk tolerance

**Panel 6: Returns Distribution**
- Should be centered near positive mean
- Check: Heavy tails? Skewness?

**Panel 7: Entry/Exit Points**
- Green triangles = entries
- Red triangles = exits
- Check: Are entries at extremes? Exits near mean?

**Panel 8: P&L by Regime**
- Compare performance in Correlated vs Divergence
- Ideally: Profitable in Correlated, breakeven/small loss in Divergence
- Check: Is regime gating working?

### Key Metrics

**Total Return**
- Absolute P&L over backtest period
- Context-dependent (depends on capital, leverage)

**Sharpe Ratio**
- Risk-adjusted return
- **Target**: > 1.0 (good), > 2.0 (excellent)
- **Warning**: < 0.5 (strategy may not be viable)

**Max Drawdown**
- Worst peak-to-trough decline
- **Target**: < 15%
- **Warning**: > 30% (too risky)

**Win Rate**
- Percentage of profitable trades
- **Target**: > 55% for mean reversion
- **Note**: Can be profitable with <50% if wins >> losses

**Total Trades**
- More trades → more confidence in statistics
- Too few (<20) → results may be luck
- Too many (>1000) → overtrading, high costs

### Red Flags

🚩 **Sharpe < 0**: Strategy is unprofitable
🚩 **Max DD > 50%**: Extremely risky
🚩 **All profit from 1-2 trades**: Luck, not skill
🚩 **Win rate < 35%**: Losses too frequent
🚩 **P&L in Divergence > Correlated**: Regime detection inverted
🚩 **Half-life > 200**: Spread doesn't revert fast enough
🚩 **Hurst > 0.6**: Spread is trending, not mean-reverting
🚩 **Z-scores consistently > ±10**: Scaling issues in data

---

## 🔧 Troubleshooting

### Problem: Strategy loses money

**Check**:
1. Mean reversion tests: Is spread actually mean-reverting?
2. Transaction costs: Are they too high relative to edge?
3. Entry/exit thresholds: Too tight (overtrading) or too wide (missed trades)?
4. Regime gating: Profitable in Correlated but losses in Divergence?

**Fix**:
- Widen entry thresholds
- Reduce trading frequency
- Improve regime detection
- Check data quality

### Problem: Too few trades

**Check**:
1. Entry threshold too high
2. Regime gating too aggressive
3. Data too short

**Fix**:
- Lower z_enter
- Reduce hysteresis
- Use more data

### Problem: High turnover, low profit

**Check**:
1. Overtrading (too many entries/exits)
2. Exit threshold too tight
3. Costs eating into profits

**Fix**:
- Widen z_enter
- Increase z_exit
- Add rollover confirmation
- Reduce position sizing

### Problem: Regime detection not working

**Check**:
1. Divergences not captured
2. False regime switches
3. Hysteresis too low/high

**Fix**:
- Try HMM instead of rules
- Adjust rolling window
- Tune hysteresis parameter
- Add more features (distance, residual vol, change points)

---

## 📈 Next Steps

1. **Load your actual data** from IMC Prosperity 3 Round 2
2. **Run mean reversion tests** on spread S
3. **Backtest with default parameters** to get baseline
4. **Analyze divergence periods** at timestamps ~12804 and ~16666
5. **Tune parameters** using sensitivity analysis
6. **Validate regime detection** aligns with observed divergences
7. **Walk-forward test** if you have enough data
8. **Generate trading signals** for live/paper trading

---

## 📚 References

**Statistical Tests**:
- Dickey & Fuller (1979): Distribution of estimators for autoregressive time series
- Hurst (1951): Long-term storage capacity of reservoirs
- Lo & MacKinlay (1988): Stock market prices do not follow random walks

**Regime Detection**:
- Hamilton (1989): A new approach to the economic analysis of nonstationary time series
- Killick et al. (2012): Optimal detection of changepoints

**Pairs Trading**:
- Gatev et al. (2006): Pairs trading: Performance of a relative-value arbitrage rule
- Vidyamurthy (2004): Pairs Trading: Quantitative Methods and Analysis

---

**Built for IMC Prosperity 3 Round 2** | Production-Ready Statistical Arbitrage System
