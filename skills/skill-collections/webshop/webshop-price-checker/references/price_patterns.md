# Common Price String Patterns
This reference lists common formats for price strings encountered in e-commerce environments. The `check_price.py` script is designed to handle these.

## Primary Patterns
*   `$164.95`
*   `Price: $164.95`
*   `$277.99`
*   `$100.0` (Note: May be displayed with one decimal place)
*   `$1,299.99` (Includes comma as thousands separator)

## Implied Context
On a product page, the price is often near:
*   The "Buy Now" button.
*   The product title.
*   A label such as "Price:", "Cost:", or "Our Price:".

## Notes for the Agent
1.  The budget constraint in the instruction may be phrased as:
    *   "price lower than 200.00 dollars"
    *   "under $200"
    *   "less than 200.00"
2.  Always convert the budget phrase into a numerical `max_budget` float (e.g., `200.00`) before calling the script.
3.  If multiple prices are present (e.g., a sale price and a list price), target the current selling price, which is usually the most prominent.
