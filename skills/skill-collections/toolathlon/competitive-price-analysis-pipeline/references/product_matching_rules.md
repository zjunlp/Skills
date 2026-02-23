# Product Matching Rules & Heuristics

## Core Principle
Match internal product names to competitor product names based on semantic similarity, shared keywords, and product category context.

## Strategies (Order of Preference)
1.  **Keyword Mapping Table**: Pre-defined mapping of internal product names to lists of keywords found in competitor product names (see `scripts/match_products.py`).
2.  **Substring Matching**: Check if the internal product name is a substring of the competitor name, or vice versa (case-insensitive).
3.  **Category Context**: Use the product category from the internal data (if available) to narrow down matches within the competitor's catalog (e.g., only match "Electronics" with competitor's "Electronics Category" products).

## Example from Trajectory
| Internal Product     | Competitor Product (Matched)          | Matching Logic                                                                 |
|----------------------|---------------------------------------|--------------------------------------------------------------------------------|
| SmartWidget Pro      | SmartWidget Professional Edition      | Keyword "SmartWidget" + Category (Electronics)                                 |
| DataFlow Analyzer    | DataFlow Business Analyzer            | Keyword "DataFlow" + "Analyzer"                                                |
| CloudSync Enterprise | CloudSync Business Suite              | Keyword "CloudSync" + Category (Cloud Services)                                |
| SecureVault Plus     | SecureVault Premium                   | Keyword "SecureVault"                                                          |
| AutoTask Manager     | AutoTask Professional                 | Keyword "AutoTask"                                                             |

## Handling Ambiguities & No Matches
- **Multiple Matches**: If an internal product matches multiple competitor products, log a warning and select the first match, or choose based on the closest price point/category.
- **No Match Found**: Log the internal product name as unmatched. Do not include it in the final CSV/BigQuery load. Report these in the final summary.
- **Price Format Variations**: Competitor prices may have currency symbols (`$`), commas (`1,399.00`), or be monthly (`129.99/month`). The extraction script must normalize these to a float.
