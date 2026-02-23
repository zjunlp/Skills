# WebShop Action Grammar Reference

## Overview
This document defines the strict grammar for interacting with the WebShop environment. Adherence is required for successful execution.

## Action Types

### 1. Search Action
*   **Format:** `search[<keywords>]`
*   **Purpose:** To query the product database.
*   **Construction Rules:**
    *   `<keywords>` is a single string.
    *   Combine the core product name with the most salient modifiers from the user instruction.
    *   **Example:** Instruction: "soft organic cotton socks for men under $20"
    *   **Query:** `search[organic cotton socks for men]`

### 2. Click Action
*   **Format:** `click[<value>]`
*   **Purpose:** To interact with any clickable element on the page.
*   **Value Source:** The `<value>` must be copied **exactly** (including case) from the `Observation` text.
*   **Common Clickable Values:**
    *   **Product IDs:** Listed next to products in search results (e.g., `B09LKXR62L`). Use the **lowercase** version for the action (e.g., `click[b09lkxr62l]`).
    *   **Navigation:** `Back to Search`, `< Prev`, `Next >`.
    *   **Page Elements:** `Description`, `Features`, `Reviews`.
    *   **Primary Action:** `Buy Now`.

## Observation Parsing Guide
The `Observation` text is structured with `[SEP]` delimiters. Key sections after a search:
1.  `Page X (Total results: Y)` - Pagination info.
2.  `Product_ID` - The clickable identifier for the product.
3.  `Product Title` - The full name/description.
4.  `$Price` - The product price.

**To click on a product, always use its `Product_ID` in lowercase.**
