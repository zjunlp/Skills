# Campaign Logic Reference

## Campaign Types & Data Requirements

### Type A: New Product Reservation Alerts
**Objective:** Notify subscribed customers about products launching soon.
- **Trigger:** User request for "new product reservations" or "upcoming releases".
- **Customer Filter:** `meta_data.subscription_preferences.new_product_alerts == True`
- **Product Filter:** `product.status IN ('draft', 'pending', 'future')` AND `product.scheduled_date <= TODAY + 30 days`
- **Audience:** Targeted (subset of customers).
- **Default Fallback:** If no subscribed customers exist, campaign has zero recipients.

### Type B: Discount Product Reminders
**Objective:** Notify all customers about products currently on sale.
- **Trigger:** User request for "discount reminders" or "promotion notifications".
- **Customer Filter:** None (all customers).
- **Product Filter:** `product.onSale == true`
- **Audience:** Broadcast (all customers).
- **Default Fallback:** If no products are on sale, campaign has no content.

## Store State Assessment Matrix
| Customers | Products (Any) | Campaign Viable? | Next Action |
| :--- | :--- | :--- | :--- |
| 0 | Any | No | Stop. Report "No customers to email." |
| >0 | 0 | No | Stop. Report "No products to promote." |
| >0 | >0 | Yes | Proceed to filter for specific campaign type. |

## Data Handling Notes
- **Pagination:** The `woocommerce-woo_*_list` tools default to 10 items per page. Use the `perPage` parameter (max typically 100) and implement a loop to fetch all data if necessary.
- **Meta Data Filtering:** The WooCommerce REST API may not support filtering customers by nested meta fields like `subscription_preferences`. You must retrieve all customers and filter the list locally.
- **Date Calculation:** "Next 30 days" is calculated from the current date. Product scheduled dates are often stored in meta fields (e.g., `_sale_price_dates_from`, `_subscription_sign_up_fee` for subscriptions, or custom fields). You may need to inspect product metadata to find the relevant date field.
