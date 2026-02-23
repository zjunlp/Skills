# BigQuery Table Schema: `new_customers`

This reference details the target table schema used for syncing new customer data from WooCommerce.

## Table Location
*   **Project:** `toolathlon-eval-ds`
*   **Dataset:** `woocommerce_crm`
*   **Table:** `new_customers`

## Column Definitions
| Column Name | Data Type | Description | Source (WooCommerce Order Field) |
| :--- | :--- | :--- | :--- |
| `customer_name` | STRING | Full name of the customer. | `billing.first_name` + `billing.last_name` |
| `email` | STRING | Customer's email address. | `billing.email` |
| `order_id` | INT64 | Unique identifier of the customer's first order. | `id` |
| `order_date` | TIMESTAMP | Date and time the order was completed (UTC). | `date_completed_gmt` |
| `order_total` | FLOAT64 | Total amount of the order. | `total` |

## Sample INSERT Query
