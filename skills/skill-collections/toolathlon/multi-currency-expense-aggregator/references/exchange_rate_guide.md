# Exchange Rate Fetching Guide

## Yahoo Finance Ticker Format for Forex
Yahoo Finance uses the format `XXXYYY=X` to represent the XXX/YYY exchange rate, where XXX is the base currency and YYY is the quote currency.

### Common Pairs for CNY Conversions
*   `USDCNY=X` - US Dollar to Chinese Yuan
*   `EURCNY=X` - Euro to Chinese Yuan
*   `GBPCNY=X` - British Pound to Chinese Yuan
*   `JPYCNY=X` - Japanese Yen to Chinese Yuan
*   `AUCNY=X`  - Australian Dollar to Chinese Yuan (Note: AUD, not AUS)
*   `CADCNY=X` - Canadian Dollar to Chinese Yuan

### Handling Indirect/Cross Rates
Some currency pairs may not have direct XXXCNY feeds. In this case, use a two-step conversion via USD:

1.  Fetch `XXXUSD=X` (e.g., `TRYUSD=X` for Turkish Lira to USD)
2.  Fetch `USDCNY=X`
3.  Calculate: `XXX_to_CNY = XXXUSD_rate × USDCNY_rate`

**Example from Trajectory:**
- TRY/CNY not found directly.
- Fetched `TRYUSD=X = 0.025468`
- Fetched `USDCNY=X = 7.2037`
- Calculated `TRY_to_CNY = 0.025468 × 7.2037 ≈ 0.1835`

## API Call Notes
*   Use the `yahoo-finance-get_stock_price_by_date` tool.
*   The `close` price from the returned data is typically used for conversion.
*   Ensure the `date` parameter matches the historical date requested by the user (e.g., "2025-06-05").
*   If "No trading data found" is returned for a direct pair, attempt the cross-rate method.
