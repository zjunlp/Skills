# Common Query Patterns for Product Search

This reference document catalogs common linguistic patterns found in shopping queries to aid in the development and refinement of the query parser.

## 1. Product Type Indicators
- **Direct Request:** "I need/want/looking for a [product]"
- **Imperative:** "Find/show me [product]"
- **Question:** "Where can I get [product]?" or "Do you have [product]?"

## 2. Attribute & Feature Patterns
- **Adjective-Noun:** "long hair", "natural looking extensions"
- **Compound Adjectives:** "clip-in", "heat-resistant", "easy-to-use"
- **Prepositional Phrases:** "with clips", "made of synthetic hair"
- **Relative Clauses:** "which is natural looking", "that costs under $40"

## 3. Constraint Patterns (Price Focus)
- **Explicit Price Limit:**
  - "price lower/less than [amount]"
  - "under/below [amount]"
  - "[amount] or less"
  - "maximum/minimum [amount]"
  - "around/about [amount]"
  - "budget of [amount]"

- **Currency Formats:**
  - "$40.00"
  - "40 dollars"
  - "40 USD"
  - "forty dollars"

## 4. Common Product Descriptors
- **Size/Length:** long, short, medium, 16-inch, 22"
- **Material:** synthetic, human hair, remy, cotton, leather
- **Color:** black, brown, blonde, red, blue, gradient
- **Style:** straight, curly, wavy, natural, professional
- **Quality:** premium, high-quality, durable, authentic

## 5. Ambiguity & Edge Cases
- **Multiple Products:** "hair extensions and clips" (parse as primary product: hair extensions)
- **Negation:** "not synthetic" (extract as feature: human hair)
- **Comparatives:** "cheaper than $50" (constraint: < 50)
- **Ranges:** "between $30 and $50" (constraint: 50, note as range in features)

## 6. Example Queries for Testing
1. "find me a wireless mouse under $25"
2. "I want organic cotton t-shirts that are machine washable"
3. "looking for a waterproof watch with GPS under $200"
4. "need a laptop backpack with usb charging port, maximum $60"
5. "show me running shoes for women, size 8, around $80"
