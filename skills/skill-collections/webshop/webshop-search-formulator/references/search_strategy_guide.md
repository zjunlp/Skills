# Search Strategy Guide for E-commerce Agents

## Attribute Priority Hierarchy
Use this hierarchy to decide which attributes to include in your initial search string (Higher = More likely to include).

1.  **Specific Variant Identifiers:** Exact size, exact color name (e.g., "patent-beige", "midnight blue"), exact model number.
2.  **Product Type/Subcategory:** The core item type (e.g., "high heel", "running shoe", "blender").
3.  **Key Material/Feature:** A primary material or feature if it's central to the request (e.g., "rubber sole", "stainless steel", "wireless"). Omit if it's a common default or likely to be tagged in metadata rather than title.
4.  **Brand or Style:** If the user specifies a brand ("Nike") or style ("wedge").
5.  **Demographic/Generic Descriptors:** Terms like "women's", "men's", "kids". Often implied or can be filtered later; include only if crucial for disambiguation.

## Common Pitfalls & Solutions
| Pitfall | Example Bad Search | Better Approach | Reason |
| :--- | :--- | :--- | :--- |
| **Over-Specification** | `search[women's us size 5 high heel shoe rubber sole patent-beige under $90]` | `search[size 5 patent-beige high heel]` | Long queries often match *fewer* items. Price and secondary features are better used as filters after the search. |
| **Under-Specification** | `search[high heel shoes]` | `search[beige high heel size 5]` | Returns too many irrelevant results, making navigation inefficient. |
| **Incorrect Order** | `search[heel high patent beige 5 size]` | `search[size 5 patent-beige high heel]` | Follows natural language query patterns that search engines are optimized for. |
| **Including Non-Keyword Filters** | `search[high heel price<90]` | `search[high heel]` (then filter by price later) | Search bars typically do not parse operators like `<`. Use the platform's built-in price filter after the search. |

## Platform Assumptions (General E-commerce)
*   **Search Syntax:** Assumes a simple keyword-matching engine. No advanced operators (e.g., `AND`, `OR`, quotes) are guaranteed to work.
*   **Result Ranking:** Results are typically ranked by relevance (keyword match in title/description) and popularity.
*   **Filtering:** Attributes like price, brand, and sometimes specific colors/materials are available as post-search filters (clickable options).

## Workflow Integration
This skill outputs the `search[keywords]` action. The subsequent agent workflow should be:
1.  Execute the `search` action from this skill.
2.  Observe the search results page.
3.  Use available `click[]` actions to:
    *   Navigate pages (`Next >`).
    *   Select specific products (by ID or name).
    *   Apply filters (size, color, price) that appear on the results page.
