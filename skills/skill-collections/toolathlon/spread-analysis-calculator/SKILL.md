---
name: spread-analysis-calculator
description: Calculates price spreads, MoM% changes, and rolling z-scores for pair trading analysis. Implements statistical normalization with configurable parameters.
---
# Skill: Spread Analysis Calculator

## Purpose
Perform statistical analysis on price spreads between two financial instruments (e.g., WTI-Brent, gold-silver). Calculate spread values, month-over-month percentage changes, and rolling z-scores for pair trading strategies.

## Core Workflow

### 1. Data Collection & Preparation
- **Input**: Two time-series datasets (e.g., WTI and Brent monthly closes)
- **Action**: Fetch data from specified sources (Yahoo Finance, CSV files, APIs)
- **Output**: Aligned, cleaned datasets with consistent timestamps

### 2. Spread Calculation
- **Formula**: `Spread = Instrument_B - Instrument_A`
- **Precision**: Round to 4 decimal places for prices, 2 decimal places for percentages

### 3. MoM% Calculation
- **Formula**: `MoM% = (Current_Value / Previous_Value - 1) × 100%`
- **Handling**: First month has `null`/`None` MoM value
- **Apply to**: Both individual instruments and the spread

### 4. Z-Score Calculation (Rolling Window)
- **Window**: Configurable (default: 6 periods)
- **Statistics**:
  - Mean: `μ = Σx / n`
  - Sample Std Dev: `σ = √[Σ(x - μ)² / (n - 1)]` (ddof=1)
  - Z-Score: `z = (x - μ) / σ`
- **Edge Cases**:
  - Return `z = 0` if window has < 4 samples
  - Return `z = 0` if standard deviation = 0
  - Clip z-scores to range [-3, 3]
- **Output**: Z-score for current period based on rolling window

### 5. Signal Generation (Optional)
- **Thresholds**: Configurable (default: ±1)
  - `z ≤ -threshold`: LONG_SPREAD (Long B + Short A)
  - `z ≥ +threshold`: SHORT_SPREAD (Short B + Long A)
  - Otherwise: FLAT
- **Regime Classification**:
  - `z ≥ +threshold`: HIGH
  - `z ≤ -threshold`: LOW
  - Otherwise: NEUTRAL

### 6. Backtest Implementation (Optional)
When backtesting is requested:
- **Position Rules**:
  - Signals generated at period-end
  - Held for 1 period, closed at next period-end
  - Only one position at a time
  - Equal weight on both legs
- **Return Calculation**:
  - For LONG_SPREAD: `Return = 0.5 × (B_return) + 0.5 × (-A_return)`
  - For SHORT_SPREAD: `Return = 0.5 × (-B_return) + 0.5 × (A_return)`
  - Include transaction costs (configurable, default: 0.40% round-trip)
- **Performance Metrics**:
  - Total Return (compounded)
  - Annualized Return
  - Sharpe Ratio (annualized)
  - Win Rate
  - Maximum Drawdown

## Output Structure
Generate analysis results in this format:
1. **Summary Table**: Period, Instrument A Close, Instrument B Close, Spread, MoM%, Z-Score, Signal, Regime
2. **Trade Log** (if backtesting): Entry/Exit dates, signals, spreads, returns, PnL
3. **Performance Metrics** (if backtesting): Total return, Sharpe, win rate, max drawdown

## Error Handling
- Validate data alignment (same timestamps, no missing periods)
- Handle division by zero in MoM% calculations
- Gracefully handle insufficient data for z-score windows
- Log warnings for edge cases

## Configuration Parameters
- `window_size`: Rolling window for z-score (default: 6)
- `z_threshold`: Signal threshold (default: 1.0)
- `clip_range`: Z-score clipping range (default: [-3, 3])
- `min_samples`: Minimum samples for z-score (default: 4)
- `transaction_cost`: Round-trip cost for backtest (default: 0.40%)
- `ddof`: Degrees of freedom for std dev (default: 1)

## Integration Notes
- Use `scripts/calculate_spread.py` for core calculations
- Refer to `references/calculation_methodology.md` for detailed formulas
- Customize output format based on destination (Notion, CSV, JSON, etc.)
