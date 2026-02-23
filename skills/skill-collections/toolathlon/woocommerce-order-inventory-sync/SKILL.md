---
name: woocommerce-order-inventory-sync
description: When the user needs to process paid WooCommerce orders to calculate raw material consumption based on Bill of Materials (BOM), update inventory levels, and sync maximum producible quantities back to WooCommerce stock. This skill monitors new paid orders, retrieves SKU and quantity data, calculates material requirements from BOM data (typically stored in Google Sheets or similar), deducts consumed materials from inventory, recalculates maximum producible product quantities based on remaining materials, and updates WooCommerce product stock levels. Triggers include WooCommerce order processing, inventory management, BOM-based material calculations, stock synchronization, and production planning workflows.
---
# Instructions

## Overview
This skill automates the end-to-end process of converting paid WooCommerce orders into raw material consumption, updating inventory, and recalculating available stock based on Bill of Materials (BOM). It connects WooCommerce with Google Sheets for BOM and inventory management.

## Prerequisites
1. **WooCommerce Connection**: Configured with API access to list orders and update products.
2. **Google Sheets Connection**: Access to a spreadsheet containing:
   - `BOM` sheet: Product_SKU, Material_Code, Material_Name, Quantity_Per_Unit
   - `Material_Inventory` sheet: Material_Code, Material_Name, Stock_Quantity
3. **Configuration File**: A `config.json` file with:
   - `spreadsheet_id`: Google Sheets ID (not folder ID)
   - `bom_sheet_name`: Name of BOM sheet (default: "BOM")
   - `inventory_sheet_name`: Name of inventory sheet (default: "Material_Inventory")
   - `product_mapping`: Mapping of product SKUs to WooCommerce product IDs

## Execution Flow

### Step 1: Monitor Paid Orders
1. Use `woocommerce-woo_orders_list` to retrieve orders with status `["processing"]` (paid orders).
2. Extract from each order:
   - `order_id`
   - `line_items`: For each item, extract `sku` and `quantity`
   - Note: The `_reduced_stock` meta indicates WooCommerce has already reduced finished product stock.

### Step 2: Retrieve BOM and Inventory Data
1. Read configuration from `config.json` to get spreadsheet ID and sheet names.
2. Use `google_sheet-get_sheet_data` to fetch:
   - BOM data from the specified BOM sheet
   - Current inventory from the specified inventory sheet
3. Validate data structure:
   - BOM must have columns: Product_SKU, Material_Code, Quantity_Per_Unit
   - Inventory must have columns: Material_Code, Stock_Quantity

### Step 3: Calculate Material Consumption
1. Aggregate total quantities needed for each product SKU across all orders.
2. For each product SKU, look up its BOM entries.
3. Calculate total material consumption:
   - `material_consumption[material_code] = sum(product_quantity × quantity_per_unit)`
4. Log detailed consumption breakdown for verification.

### Step 4: Update Raw Material Inventory
1. For each material in inventory:
   - Subtract consumed quantity from current stock
   - Calculate new stock level
2. Use `google_sheet-update_cells` to write updated inventory back to Google Sheets.
3. Maintain the same data structure (Material_Code, Material_Name, Stock_Quantity).

### Step 5: Calculate Maximum Producible Quantities
1. For each product SKU:
   - For each material in its BOM:
     - Calculate `max_from_material = floor(available_stock / quantity_per_unit)`
   - Determine `min(max_from_material)` across all materials → maximum producible quantity
   - Identify the limiting material (bottleneck)
2. This calculation determines the actual stock that can be promised/sold.

### Step 6: Sync to WooCommerce Stock
1. Use `woocommerce-woo_products_batch_update` to update product stock quantities.
2. Update `stock_quantity` for each product based on maximum producible calculation.
3. Ensure `manage_stock: true` is set for each product.

### Step 7: Verification and Reporting
1. Fetch updated inventory from Google Sheets to confirm changes.
2. Fetch updated products from WooCommerce to confirm stock levels.
3. Provide a comprehensive summary including:
   - Orders processed
   - Material consumption
   - Updated inventory levels
   - Maximum producible quantities
   - Updated WooCommerce stock levels

## Error Handling
1. **Missing Configuration**: Check `config.json` exists and has required fields.
2. **Invalid Spreadsheet ID**: Verify it's a spreadsheet ID, not a folder ID.
3. **Missing Sheets**: Verify BOM and inventory sheets exist with correct names.
4. **Data Validation**: Check for missing SKUs in BOM, negative inventory, etc.
5. **API Errors**: Handle WooCommerce and Google Sheets API errors gracefully.

## Notes
1. The skill assumes WooCommerce has already reduced finished product stock via `_reduced_stock` meta.
2. Decimal quantities in BOM (e.g., 0.5L of varnish) are supported.
3. The limiting material calculation uses integer division (floor) for practical production planning.
4. Consider running this skill on a schedule (e.g., hourly) or triggered by order status changes.
