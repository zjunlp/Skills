# Reference: Common E-commerce Purchase UI Patterns

This document catalogs common button text and UI patterns for purchase actions across e-commerce platforms. Use this to improve the robustness of the purchase button detection logic.

## Primary Purchase Buttons
- **Buy Now**: The most direct purchase action. Often leads to a single-page checkout.
- **Add to Cart**: Places item in shopping cart. May require a subsequent "Checkout" action.
- **Purchase**: A generic term, common on some marketplaces.
- **Checkout**: Typically appears on cart pages or as a combined "Add to Cart & Checkout" button.

## Secondary/Contextual Actions
- **Subscribe and Save**: For recurring purchases.
- **Rent Now**: For rental services.
- **Pre-order**: For items not yet released.
- **Bid/Buy It Now**: On auction sites.

## Stylistic Variations & Casing
Buttons may appear in different casings:
- `Buy Now`
- `BUY NOW`
- `buy now`
- `Add To Cart`

## Localization Examples
- **French**: `Acheter maintenant`, `Ajouter au panier`
- **Spanish**: `Comprar ahora`, `Añadir al carrito`
- **German**: `Jetzt kaufen`, `In den Warenkorb`

## Implementation Note for the Skill
The `execute_purchase.py` script prioritizes "Buy Now" patterns. If the target platform uses different terminology, the skill's text instructions should guide the agent to identify the correct button based on the specific UI context provided in the observation.
