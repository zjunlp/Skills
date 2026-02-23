---
name: inventory-monitoring-and-alert-system
description: When the user needs to monitor product inventory levels, check stock quantities against safety thresholds, identify low-stock items, and trigger automated notifications and documentation updates. This skill is triggered by keywords like 'inventory levels', 'stock quantity', 'safety threshold', 'low-stock', 'purchase requisition', 'stock alert', or when working with e-commerce platforms like WooCommerce. It provides capabilities to 1) Retrieve product data from e-commerce systems, 2) Analyze stock levels against configurable thresholds, 3) Identify products requiring replenishment, 4) Update Google Sheets or other documentation systems with low-stock items, and 5) Send automated email notifications to purchasing managers using customizable templates.
---
# Inventory Monitoring and Alert System

## Overview
This skill automates inventory monitoring by checking product stock levels against safety thresholds, updating purchase requisition lists, and sending email alerts.

## Core Workflow

### 1. Initial Setup & Data Collection
- **Retrieve Products**: Fetch all products from WooCommerce using `woocommerce-woo_products_list`.
- **Locate Resources**: 
  - Find the Google Sheets purchase requisition list using `google_sheet-list_spreadsheets`.
  - Read the purchasing manager's email address from `purchasing_manager_email.txt`.
  - Load the email template from `stock_alert_email_template.md`.

### 2. Inventory Analysis
- For each product, compare `stock_quantity` against `stock_threshold` (stored in `meta_data`).
- Identify all products where `stock_quantity < stock_threshold`.
- Extract supplier information from product meta_data (name, contact, ID).

### 3. Documentation Update
- Access the Google Sheet using the spreadsheet ID.
- Check existing sheet structure using `google_sheet-get_sheet_data`.
- Update cells with low-stock product information:
  - Product ID, Name, SKU
  - Current Stock, Safety Threshold
  - Supplier Name, Supplier ID, Supplier Contact
  - Alert Time (current date), Suggested Order Quantity (threshold × 1.5)

### 4. Notification Delivery
- For each low-stock product, generate personalized email using the template.
- Send email to purchasing manager with:
  - Product-specific details
  - Stock vs. threshold comparison
  - Supplier contact information
  - Link to Google Sheets requisition list

## Key Considerations
- **Pagination**: Handle large product lists by adjusting `perPage` parameter.
- **Error Handling**: Verify all required files exist before processing.
- **Data Consistency**: Ensure Google Sheets formatting matches expected columns.
- **Email Personalization**: Customize each email with specific product details.

## Success Criteria
- All low-stock products identified and documented
- Google Sheets updated with complete product information
- Individual email alerts sent for each low-stock item
- Clear summary provided to user with counts and details
