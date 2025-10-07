# imc-prosperity3-round2
# Basket Statistical Arbitrage System
### Production-Ready Implementation for IMC Prosperity 3 Round 2
#### Built in Claude

---

## 📦 What's Included

This is a **complete, production-grade statistical arbitrage system** with:

✅ **3 Python Modules** (~1,500 lines of clean, tested code)
- `basket_stat_arb.py` - Core trading engine
- `basket_arb_viz.py` - Visualization & analysis tools
- `basket_arb_quickstart.py` - Copy-paste quick start

✅ **Comprehensive Documentation**
- `basket_arb_guide.md` - Full usage guide with 8 examples
- This README - Quick overview

✅ **Key Features**
- Deterministic basket replication (no model error)
- Complete mean reversion testing suite
- Algorithmic divergence detection
- Regime-aware signal generation
- Realistic backtesting with costs
- Parameter optimization tools

---

## 🚀 Quick Start (3 Minutes)

### 1. Install Dependencies
```bash
pip install numpy pandas scipy statsmodels matplotlib seaborn
pip install ruptures hmmlearn  # Optional for advanced features
```

### 2. Run the Quick Start Script
```python
# Just run this - it does everything:
python basket_arb_quickstart.py
```

**That's it!** The script will:
- Load your data
- Test mean reversion
- Detect divergence periods algorithmically
- Run backtest
- Generate visualizations
- Export trading signals

### 3. Or Use Interactively
```python
import pandas as pd
from basket_stat_arb import backtest, BacktestConfig
from basket_arb_viz import analyze_divergence_periods

# Load data
prices = pd.read_csv('your_data.csv')[['C', 'J', 'D']]

# Detect divergences (answers your original question!)
analysis = analyze_divergence_periods(prices, BacktestConfig())
print(f"Detected {len(analysis['divergence_periods'])} divergence periods")

# Run backtest
results = backtest(prices, BacktestConfig())
print(f"Sharpe: {results['metrics']['sharpe_ratio']:.2f}")
```

---

## 🎓 The Core Concept

### The Deterministic Replication Identity

Your baskets have a **mathematical relationship** (not statistical):

```
B1 = 6×C + 3×J + 1×D
B2 = 4×C + 2×J

Therefore: B1 = 1.5×B2 + D  (exactly, always)
```

**Mispricing Spread**:
```
S = B1 - (1.5×B2 + D)
```

In a perfect market, `S = 0`. Any deviation is a **pure pricing error** (not model uncertainty).

This is **stronger than cointegration** because:
- ✅ No parameter estimation error
- ✅ No hedge ratio uncertainty
- ✅ Exact replication (not statistical)
- ✅ Deviations are guaranteed mispricings

---

## 🔍 Answering Your Original Question

### "How do I detect divergences algorithmically?"

You observed visual divergences around timestamps 12804 and 16666. The system detects these **automatically** using:

**Method 1: Use the built-in function (Recommended)**
```python
from basket_arb_viz import analyze_divergence_periods, print_divergence_report

analysis = analyze_divergence_periods(prices, config)
print_divergence_report(analysis)

# Get exact timestamps
for start, end in analysis['divergence_periods']:
    print(f"Divergence: {start} to {end}")
```

**Method 2: Understand the detection criteria**

The system flags divergence when **multiple indicators** trigger:

1. **Correlation Breakdown**
   - Fisher z-transformed correlation CI crosses zero
   - Indicates C-J inverse relationship weakening

2. **High Residual Volatility**
   - Component pair hedge becomes unstable
   - |z-score of residual| > 2.5

3. **Large Euclidean Distance**
   - Normalized C and inverted J series diverge
   - Distance > 80th percentile

4. **Hysteresis Filter**
   - Requires 5+ consecutive bars to confirm
   - Prevents false switches on noise

**Method 3: Change-point detection**
```python
from basket_arb_viz import detect_change_points

# Find structural breaks in the spread
change_points = detect_change_points(df['S'], method='pelt', penalty=10.0)
print(f"Structural breaks at: {change_points}")
```

These methods will tell you **exactly** where divergences occur based on statistical criteria, not visual inspection.

---

## 📊 Key Outputs

### 1. Mean Reversion Tests
Tells you if the spread is tradeable:
- **ADF Test**: Is it stationary? (p < 0.10 ✓)
- **Hurst Exponent**: Is it mean-reverting? (H < 0.5 ✓)
- **Variance Ratio**: Does structure support MR? (VR < 1 ✓)
- **Half-life**: How fast does it revert? (< 100 bars ✓)

### 2. Divergence Analysis
Shows when C-J relationship breaks:
- **Timestamps** of each divergence period
- **Duration** of each period
- **Percentage** of time in divergence
- **Visual confirmation** via plots

### 3. Backtest Results
Evaluates strategy profitability:
- **Sharpe Ratio** (target > 1.0)
- **Max Drawdown** (target < 15%)
- **Win Rate** (target > 55%)
- **P&L by Regime** (should be positive in Correlated)

### 4. Trading Signals
Ready-to-execute positions:
- **Entry/Exit** timestamps
- **Position size** (arb units)
- **Leg weights** (B1, B2, D)
- **Regime state** (Correlated vs Divergence)

---

## 🎛️ Parameter Tuning Guide

### Start Here (Defaults)
```python
config = BacktestConfig(
    z_enter=2.0,      # Enter at 2 std devs
    z_exit=0.5,       # Exit at 0.5 std devs
    halflife_mu=50,   # 50-bar EWMA for mean
    halflife_sigma=50 # 50-bar EWMA for vol
)
```

### If Too Few Trades
- ⬇️ Lower `z_enter` to 1.5
- ⬇️ Reduce `hysteresis_bars` to 3

### If Too Many Trades (High Costs)
- ⬆️ Raise `z_enter` to 2.5 or 3.0
- ⬆️ Increase `z_exit` to 0.7

### If Missing Divergences
- ⬇️ Shorten `rolling_window` to 30
- ⬇️ Lower `fisher_alpha` to 0.01
- ⬆️ Check distance threshold

### If Too Many False Divergences
- ⬆️ Lengthen `rolling_window` to 70
- ⬆️ Increase `hysteresis_bars` to 7

---

## 📈 Typical Results

With the default configuration on mean-reverting data:

| Metric | Good | Excellent |
|--------|------|-----------|
| Sharpe Ratio | > 1.0 | > 2.0 |
| Max Drawdown | < 20% | < 10% |
| Win Rate | > 55% | > 65% |
| Divergence Time | 10-20% | 5-15% |

---

## 🛠️ File Structure

```
basket_arbitrage_system/
│
├── basket_stat_arb.py          # Core engine (500 lines)
│   ├── Basket construction
│   ├── Mean reversion tests
│   ├── Regime detection
│   ├── Signal generation
│   └── Backtest engine
│
├── basket_arb_viz.py           # Visualization (600 lines)
│   ├── Plotting functions
│   ├── Parameter sensitivity
│   ├── Divergence analysis
│   ├── Change-point detection
│   └── HMM regime detection
│
├── basket_arb_quickstart.py    # Quick start script (400 lines)
│   └── End-to-end workflow
│
├── basket_arb_guide.md         # Full documentation
│   ├── 8 usage examples
│   ├── Parameter guide
│   ├── Interpretation guide
│   └── Troubleshooting
│
└── README.md                    # This file
```

---

## 📚 Documentation

### For Different Use Cases

**👨‍💻 "I just want it to work"**
→ Run `basket_arb_quickstart.py`

**🔬 "I want to understand the math"**
→ Read `basket_arb_guide.md` - Core Concepts section

**📊 "I need to tune parameters"**
→ Read `basket_arb_guide.md` - Parameter Tuning section

**🐛 "Something's not working"**
→ Read `basket_arb_guide.md` - Troubleshooting section

**🎓 "I want to learn the code"**
→ Read inline docstrings in `basket_stat_arb.py`

---

## 🔧 Advanced Features

### Walk-Forward Optimization
```python
# Split data into train/test
train = prices[:int(len(prices)*0.7)]
test = prices[int(len(prices)*0.7):]

# Optimize on train
param_grid = {'z_enter': [1.5, 2.0, 2.5], 'z_exit': [0.3, 0.5, 0.7]}
sweep = run_parameter_sweep(train, config, backtest, param_grid)
best_params = sweep.iloc[sweep['sharpe_ratio'].idxmax()]

# Test on holdout
config.z_enter = best_params['z_enter']
config.z_exit = best_params['z_exit']
results = backtest(test, config)
```

### HMM Regime Detection
```python
# Build features
features = build_regime_features(prices)

# Fit HMM
regime_hmm, model = classify_regime_hmm(features, n_states=2)

# Use in backtest
config.regime_method = 'hmm'  # Instead of 'rules'
```

### Custom Cost Model
```python
config.commission_bps = 5.0      # 5 bps commission
config.half_spread_bps = 10.0    # 10 bps spread
config.slippage_factor = 1.0     # Higher slippage
```

---

## ⚠️ Important Notes

### Data Requirements
- **Minimum**: 500 rows for reliable statistics
- **Recommended**: 1000+ rows
- **Columns**: Must have C (CROISSANTS), J (JAMS), D (DJEMBES)
- **Format**: Continuous timestamp index

### What This System Does NOT Do
- ❌ Real-time data ingestion (you provide historical data)
- ❌ Live order execution (outputs signals for you to execute)
- ❌ Risk management beyond position limits (you set your limits)
- ❌ Predict future prices (it trades mean reversion, not prediction)

### What This System DOES Do
- ✅ Detects divergence periods algorithmically
- ✅ Tests if spread is tradeable (mean-reverting)
- ✅ Generates entry/exit signals with regime awareness
- ✅ Backtests with realistic costs and risk controls
- ✅ Optimizes parameters
- ✅ Provides detailed performance analytics

---

## 🎯 Common Questions

**Q: Why is Sharpe ratio negative/low?**
A: Spread may not be mean-reverting. Check `mean_reversion_tests()` results. If Hurst > 0.5 or half-life > 200, spread is not tradeable.

**Q: Why aren't divergences being detected?**
A: Try `analyze_divergence_periods()` with default config first. If still nothing, lower `hysteresis_bars` or shorten `rolling_window`.

**Q: System detects divergence but I don't see it visually**
A: Statistical criteria may be more sensitive than visual inspection. Check correlation CI plot - if it crosses zero, that's divergence even if subtle.

**Q: Too many trades, eating up profit in costs**
A: Raise `z_enter` threshold and/or increase `hysteresis_bars` to reduce frequency.

**Q: How do I know if my parameters are optimal?**
A: Run `run_parameter_sweep()` and look for highest Sharpe ratio. Also check that max drawdown is acceptable.

---

## 📞 Support

**For bugs or questions about the code:**
- Check inline docstrings
- Read `basket_arb_guide.md` Troubleshooting section
- Review example code in the guide

**For strategy questions:**
- The system is framework-agnostic
- You can modify signal logic in `generate_signals()`
- You can add custom risk controls in `backtest()`

---

## 🏆 Credits

Built for **IMC Prosperity 3 Round 2** based on:
- Modern pairs trading literature
- Production trading system best practices
- Statistical arbitrage frameworks from leading hedge funds

**Key References:**
- Gatev et al. (2006) - Pairs Trading
- Lo & MacKinlay (1988) - Variance Ratio Test
- Hamilton (1989) - Regime Switching Models

---

## ✅ Quick Validation Checklist

Before trading:
- [ ] Mean reversion tests pass (ADF p < 0.10, Hurst < 0.5)
- [ ] Half-life < 100 bars
- [ ] Backtest Sharpe > 1.0
- [ ] Max drawdown < 20%
- [ ] Divergence detection validated against visual inspection
- [ ] Parameter sensitivity tested
- [ ] Realistic transaction costs included

---

**Ready to start? Run `basket_arb_quickstart.py` now!** 🚀

*Last Updated: For IMC Prosperity 3 Round 2 | Production-Ready Implementation*
