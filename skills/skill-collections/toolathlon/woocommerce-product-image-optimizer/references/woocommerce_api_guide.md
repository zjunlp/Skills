# WooCommerce API Reference Guide

## API Endpoints Used

### 1. List Products
**Endpoint:** `GET /wp-json/wc/v3/products`
**Purpose:** Fetch all published variable products
**Parameters:**
- `per_page`: Number of products per page (default: 100)
- `status`: Product status (use "publish")
- `type`: Filter by product type (optional)

**Response Structure:**
