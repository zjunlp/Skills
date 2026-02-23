---
title: Multi-Market Portfolio Allocation Workflow
---
flowchart TD
    A[User Request<br>+ Excel File] --> B[Read Excel &<br>Extract Stock List]
    B --> C[Categorize Stocks by Market<br>US / HK / A-Share]
    C --> D{Fetch Financial Data}
    D --> E[Fetch Latest Stock Prices]
    D --> F[Fetch Exchange Rates<br>USD/CNY & USD/HKD]
    E & F --> G[Calculate Allocation & Shares]
    
    subgraph G [Core Calculation]
        G1[Apply Ratio to Total Capital]
        G2[Convert to Local Currency]
        G3[Equal Weight per Stock in Market]
        G4[Floor Shares US/HK<br>Floor to Lots A-Share]
        G5[Summarize Costs & Totals]
    end

    G --> H[Generate Detailed<br>Position-Building Plan]
    G --> I[Update Excel File with<br>Tickers & Share Quantities]
    H --> J[Deliver Final Plan &<br>Confirm File Update]
    I --> J
    J --> K[Task Complete]
