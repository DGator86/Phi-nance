# Combined Strategies Guide

## Overview

The Strategy Lab now supports **combining multiple strategies** into a single backtest using voting logic. Instead of running strategies separately and comparing results, you can combine them to create a more robust trading system.

## How It Works

When you select multiple strategies and choose "Combine strategies", the system:

1. **Gets signals from each strategy** at each bar/day
2. **Combines signals** using your chosen voting mode
3. **Executes trades** based on the combined decision

## Voting Modes

### 1. Majority Vote (Default)
- **Buy**: When more than half of strategies signal buy
- **Sell**: When more than half of strategies signal sell
- **Best for**: Balancing multiple signals, reducing false positives

### 2. Unanimous
- **Buy**: Only when ALL strategies agree to buy
- **Sell**: Only when ALL strategies agree to sell
- **Best for**: Very conservative trading, only taking high-confidence trades

### 3. Weighted Average
- **Buy**: When average signal strength > 0.3
- **Sell**: When average signal strength < -0.3
- **Best for**: Smoothing signals, reducing noise

## How to Use

### In the GUI:

1. **Select multiple strategies** (checkboxes in sidebar)
2. **Choose "Combine strategies"** (radio button under "Run mode")
3. **Pick a voting mode** (dropdown appears when combining)
4. **Set symbol and dates**
5. **Click "Run backtest"**

### Example:

- Select: "Buy & Hold" + "SMA Crossover"
- Mode: "Combine strategies"
- Voting: "Majority Vote"
- Result: One combined strategy that buys when both agree, sells when both agree, or holds when they disagree

## When to Use Combined Strategies

✅ **Use combined strategies when:**
- You want to reduce false signals
- Multiple strategies complement each other
- You want a more robust system
- You're testing ensemble methods

❌ **Use individual comparison when:**
- You want to see which single strategy performs best
- You're testing strategy parameters
- You want to understand individual strategy behavior

## Technical Details

### Signal Generation

Each strategy generates signals:
- **1** = Buy signal
- **-1** = Sell signal  
- **0** = Hold/Neutral

### Signal Combination

The combined strategy aggregates these signals:
- **Majority**: Counts votes, requires >50% agreement
- **Unanimous**: Requires 100% agreement
- **Weighted**: Averages signal values, uses thresholds

### Performance

Combined strategies often show:
- **Lower trade frequency** (more selective)
- **Higher win rate** (better signals)
- **Different risk/return profile** (more stable)

## Example Results

**Individual Strategies:**
- Buy & Hold: 8.5% return, 0.8 Sharpe
- SMA Crossover: 12.3% return, 1.2 Sharpe

**Combined (Majority Vote):**
- Buy & Hold + SMA Crossover: 10.1% return, 1.1 Sharpe, fewer trades

The combined strategy may not always outperform the best individual strategy, but it often provides:
- More consistent performance
- Lower drawdowns
- Better risk-adjusted returns

## Tips

1. **Start with Majority Vote** - it's the most balanced
2. **Try different combinations** - some strategies work better together
3. **Compare results** - run both "Compare individually" and "Combine strategies" to see which works better
4. **Use Unanimous for conservative** - only trade when all strategies agree
5. **Experiment** - different voting modes suit different market conditions

## Limitations

- Currently supports strategies available in the GUI (Buy & Hold, SMA Crossover, Phi-nance Projection)
- Voting modes are fixed (can't customize thresholds yet)
- Each strategy gets equal weight (no performance-based weighting yet)

## Future Enhancements

Potential improvements:
- Performance-based weighting (weight strategies by their historical Sharpe Ratio)
- Custom thresholds for weighted mode
- More voting modes (e.g., "At least N strategies agree")
- Strategy-specific weights in UI
- Dynamic voting based on market conditions
