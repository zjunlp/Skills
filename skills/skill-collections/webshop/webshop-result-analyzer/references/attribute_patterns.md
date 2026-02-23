# Common Attribute Patterns for WebShop Analysis

Use these regular expression patterns to identify key attributes in product titles and descriptions.

## Size Patterns
- **US Size 5:** `r"\bsize\s*5\b|\b5\s*us\b|\b5\.0\b"`
- **Size Range (e.g., 5-13):** `r"\b5\s*[-–]\s*13\b"` (will match "5-13", "5 – 13")

## Color & Material Patterns
- **Patent Beige:** `r"patent-beige|beige.*patent|patent.*beige|beige-almond toe-patent leather"`
- **Rubber Sole:** `r"rubber\s*sole|sole\s*rubber"`

## Price Extraction
- **Single Price:** `\$\d+\.?\d*` (e.g., $54.99)
- **Price Range:** `\$\d+\.?\d*\s*to\s*\$\d+\.?\d*` (e.g., $49.99 to $54.99)

## General Keyword Matching
When the user provides specific terms (e.g., "high heel", "wedges"), perform a simple case-insensitive substring check. No regex is needed for simple keywords.

## Usage Note
These patterns are examples. You may need to adjust them based on the observed text formatting in the WebShop environment. The bundled script uses these as configurable inputs.
