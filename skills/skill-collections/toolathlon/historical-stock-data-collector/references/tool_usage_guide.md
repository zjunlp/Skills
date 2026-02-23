# Tool Usage Guide for Historical Stock Data

## Core Tools

### 1. yahoo-finance-get_historical_stock_prices
Retrieves daily stock price data.

**Parameters:**
- `ticker` (string, required): The stock symbol (e.g., "AAPL").
- `start_date` (string, required): Start date in `YYYY-MM-DD` format.
- `end_date` (string, required): End date in `YYYY-MM-DD` format.
- `interval` (string, required): Use `"1d"` for daily data.
- `auto_adjust` (boolean): `false` for original prices (default), `true` for prices adjusted for splits/dividends.

**Example Call:**
