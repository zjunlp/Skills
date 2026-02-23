# WooCommerce Product Data Structure

## Key Fields for Inventory Monitoring

### Product Object Fields
- `id`: Product ID (integer)
- `name`: Product name (string)
- `sku`: Stock Keeping Unit (string)
- `manage_stock`: Boolean indicating if stock is managed
- `stock_quantity`: Current stock level (integer, nullable)
- `stock_status`: Current status ("instock", "outofstock", etc.)

### Meta Data Fields (in `meta_data` array)
- `stock_threshold`: Safety threshold value (string, stored in meta)
- `supplier`: Supplier information object containing:
  - `name`: Supplier name
  - `contact`: Supplier email/contact
  - `supplier_id`: Supplier identifier
- `category`: Product category (optional)

## Accessing Meta Data
Threshold and supplier data are stored in the `meta_data` array. Example access pattern:
