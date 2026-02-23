# Memory Graph Transaction Schema

The skill reads transaction data from a memory/knowledge graph. The expected structure (based on the trajectory) is as follows:

## Entity Types
1.  **Product**: `entityType: "Product"`. Contains observations about products.
2.  **Customer**: `entityType: "Customer"`. Contains observations about customers.
3.  **Supplier**: `entityType: "Supplier"`. Contains observations about suppliers.
4.  **Purchase Transaction**: `entityType: "Purchase Transaction"`. Represents a stock purchase.
5.  **Sales Transaction**: `entityType: "Sales Transaction"`. Represents a product sale.

## Key Transaction Entity Structure
