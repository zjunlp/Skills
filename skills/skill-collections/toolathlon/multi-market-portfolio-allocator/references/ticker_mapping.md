# Common Stock Ticker Reference

This is a non-exhaustive reference for mapping common company names to their Yahoo Finance tickers across US, Hong Kong, and A-Share markets. Always prioritize the user's explicit instructions.

## US Stocks (NYSE/NASDAQ)
- Microsoft: MSFT
- Apple: AAPL
- NVIDIA: NVDA
- AMD: AMD
- Google / Alphabet: GOOGL (Class A) or GOOG (Class C)
- Meta (Facebook): META
- Amazon: AMZN
- Tesla: TSLA
- Netflix: NFLX

## Hong Kong Stocks (Suffix: .HK)
- Tencent: 0700.HK
- Alibaba: 9988.HK (Primary Listing)
- Meituan: 3690.HK
- Xiaomi: 1810.HK
- JD.com: 9618.HK
- HSBC: 0005.HK
- AIA: 1299.HK

## A-Shares
**Shanghai Stock Exchange (Suffix: .SS)**
- Kweichow Moutai: 600519.SS
- Ping An Insurance: 601318.SS
- Industrial and Commercial Bank of China (ICBC): 601398.SS
- WuXi AppTec: 603259.SS

**Shenzhen Stock Exchange (Suffix: .SZ)**
- Contemporary Amperex Technology (CATL): 300750.SZ
- BYD: 002594.SZ
- Midea Group: 000333.SZ
- Gree Electric: 000651.SZ

## Important Notes
1.  **Dual-Listings:** Some companies are listed in multiple markets.
    - Alibaba: US (BABA), Hong Kong (9988.HK). The user may specify which one to use.
    - JD.com: US (JD), Hong Kong (9618.HK).
2.  **Yahoo Finance Format:** Always use the correct suffix (.HK, .SS, .SZ) for non-US stocks.
3.  **Validation:** If unsure, verify the ticker by searching on Yahoo Finance or a similar financial data portal before fetching prices.
