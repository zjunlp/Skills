# Query Component Reference

This document lists common patterns and components found in WebShop user instructions to aid in parsing and criteria formulation.

## 1. Product Type Indicators
Look for these phrases followed by the product name:
*   "I need..."
*   "I want..."
*   "I'm looking for..."
*   "Find me..."
*   "Show me..."
*   "Buy..."

## 2. Common Attributes & Keywords
| Category | Examples |
| :--- | :--- |
| **Dietary** | gluten free, vegan, vegetarian, organic, non-gmo, kosher, halal, nut free, dairy free, sugar free, low carb, keto, paleo |
| **Health** | low calorie, high fiber, low sodium, heart healthy, probiotic, antioxidant |
| **Quality** | gourmet, premium, luxury, best selling, top rated, award winning |
| **Form/Packaging** | individual bags, bulk, pack of [number], family size, travel size, reusable container |
| **Flavor/Type** | original, chocolate, caramel, cheese, butter, spicy, sweet, salted |

## 3. Price Constraint Patterns
| Pattern | Example | Interpretation |
| :--- | :--- | :--- |
| **Upper Limit** | "lower than $X", "less than X dollars", "under X", "below X", "maximum X" | `price_max = X` |
| **Lower Limit** | "above $X", "more than X", "over X", "minimum X" | `price_min = X` |
| **Range** | "between X and Y dollars", "X - Y", "from X to Y" | `price_min = X`, `price_max = Y` |
| **Exact** | "around X", "about X" | Consider `price_max = X * 1.1`, `price_min = X * 0.9` |

## 4. Other Specifications
*   **Brand:** "SkinnyPop", "Kellogg's", "Nike"
*   **Size/Quantity:** "12 pack", "5 ounce", "1 gallon"
*   **Color:** "red", "blue", "black"
*   **Material:** "stainless steel", "cotton", "plastic"

## 5. Search Strategy Tips
*   **Primary Search:** Combine the core `product` with the most critical `attribute` (e.g., `search[gluten free popcorn]`).
*   **Secondary Filters:** If results are poor, relax less critical attributes one by one.
*   **Price Checking:** The website often lacks a price filter action. You must:
    1.  Scan the price listed next to each search result item.
    2.  Click into promising items and check the `Price:` field on the product page.
    3.  Manually reject items that violate the `price_max` or `price_min` constraints.
