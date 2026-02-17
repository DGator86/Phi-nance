# Phi-nance Strategy Lab v3 — Visual Workflow Builder

## Overview

A visual drag-and-drop interface for building trading strategy workflows. Drag strategy blocks onto a grid workspace, connect them with arrows, configure parameters, and run backtests.

## Features

### 1. Strategy Store (Right Sidebar)
- Click strategy buttons to add blocks to the workspace
- All strategies from `STRATEGY_CATALOG` are available
- Special "Composer" block for combining multiple strategies

### 2. Workspace (Main Area)
- **Grid-based layout** with drag-and-drop support (HTML component)
- **Fallback grid view** using Streamlit native widgets
- Blocks lock in place after placement
- Visual connections between blocks show execution flow

### 3. Strategy Blocks
- **Expandable**: Click ⚙️ to configure parameters
- **Movable**: Drag blocks to reposition
- **Connectable**: Click connection points to link strategies
- **Removable**: Click × to delete

### 4. Composer Block
- Special block that accepts multiple strategy inputs
- Configurable combination method:
  - **Majority Vote**: Most strategies agree
  - **Weighted**: Weighted average of signals
  - **Unanimous**: All strategies must agree
- Routes multiple strategies into a single output

### 5. Backtest Configuration
- **Ticker selection**: Enter symbol (e.g., SPY)
- **Backtest type**:
  - **Date Range**: Select start and end dates
  - **# Bars**: Specify number of bars to test
- **Timeframe**: 1m, 5m, 15m, 30m, 1h, 1d
- **Run button**: Executes backtest based on workflow

### 6. Results Display
- Shows backtest metrics in a dataframe
- Displays results after successful execution

## Usage

### Basic Workflow

1. **Add strategies**: Click strategy buttons in the right sidebar
2. **Configure parameters**: Expand each block and adjust settings
3. **Connect strategies** (optional):
   - Click output connection point (right side) of a strategy
   - Click input connection point (left side) of Composer or another strategy
4. **Set backtest config**: Expand "Backtest Configuration" panel
5. **Run backtest**: Click "Run Backtest" button
6. **View results**: Results appear below the configuration panel

### With Composer

1. Add multiple strategy blocks
2. Add a Composer block
3. Connect strategy outputs → Composer inputs
4. Configure Composer method (majority/weighted/unanimous)
5. Run backtest — Composer combines all connected strategies

## Files

- **`app_v3.py`**: Main Streamlit application
- **`components/workflow_builder/workflow_builder.html`**: HTML/JS drag-and-drop component
- **`.streamlit/config.toml`**: Dark theme configuration

## Running

```bash
streamlit run app_v3.py
```

## Technical Details

### Workflow State
Stored in `st.session_state.workflow`:
```python
{
    "blocks": [
        {
            "id": "block_0",
            "type": "sma_cross",
            "x": 50,
            "y": 50,
            "params": {"fast_period": 10, "slow_period": 20}
        }
    ],
    "connections": [
        {"from_id": "block_0", "to_id": "block_1"}
    ]
}
```

### Execution Logic
- **Single strategy**: Runs individual backtest
- **Multiple strategies → Composer**: Runs combined backtest with composer method
- **Multiple strategies (no composer)**: Runs each individually and compares

## Future Enhancements

- [ ] Full drag-and-drop with HTML5 API
- [ ] Visual connection arrows in grid view
- [ ] Save/load workflows
- [ ] Strategy execution order visualization
- [ ] Real-time parameter preview
- [ ] Workflow templates
