---
name: woocommerce-product-image-optimizer
description: When the user wants to optimize WooCommerce product images based on sales performance data, particularly for variable products with multiple variations. This skill analyzes order history to identify the best-selling variation for each product and automatically updates the main product image to showcase the most popular option. It handles fetching products and orders data, analyzing sales patterns, identifying variation popularity, and updating product images through WooCommerce API. Use when users mention 'best-selling variation', 'update main product image based on sales', 'product image optimization', or need to align product visuals with customer preferences based on order data.
---
# WooCommerce Product Image Optimizer

## Overview
This skill analyzes WooCommerce order data to identify the best-selling variations for variable products and updates the main product image to showcase the most popular option. This data-driven approach helps improve conversions by aligning product visuals with customer preferences.

## Prerequisites
- WooCommerce API access with appropriate permissions
- Variable products with variations that have distinct images
- Order history data (completed/processing orders)

## Workflow

### 1. Gather Data
- Fetch all published variable products using `woocommerce-woo_products_list`
- Fetch recent orders (completed/processing) using `woocommerce-woo_orders_list`
- For each variable product, fetch its variations using `woocommerce-woo_products_variations_list`

### 2. Analyze Sales Data
- Parse order line items to extract variation sales data
- Count total units sold per variation
- Identify the best-selling variation for each parent product
- Map variation IDs to their corresponding image data

### 3. Update Product Images
- For each product with a best-selling variation:
  - Extract the variation's image ID and URL
  - Update the parent product's main image using `woocommerce-woo_products_update`
  - Set the images array to contain only the best-selling variation's image

## Key Considerations
- Only processes variable products (type: "variable")
- Considers both "completed" and "processing" order statuses
- Handles multiple quantity purchases in orders
- In case of tie in sales, the first variation with highest sales will be selected
- The update replaces all existing product images with the single best-selling variation image

## Error Handling
- Skip products without variations
- Skip variations without images
- Log any API errors during updates
- Continue processing other products if one fails

## Output
- Summary table showing product updates
- Confirmation of successful image changes
- Details of best-selling variations and units sold
