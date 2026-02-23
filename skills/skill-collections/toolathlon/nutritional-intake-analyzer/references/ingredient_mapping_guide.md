# Ingredient Mapping Guide

## Common Challenges & Solutions

### 1. Name Variations
Ingredients may have different names in recipes vs. nutrition databases.

| Recipe Name | Database Name | Action |
|-------------|---------------|--------|
| 瘦猪肉 | 瘦肉（猪肉） | Use partial matching or manual lookup |
| 植物油 | 植物油 | Exact match |
| 小葱 | 葱 | Map to closest available |
| 料洒 | (Not in DB) | Skip if negligible, estimate if significant |

### 2. Quantity Interpretation

**Explicit Quantities:**
- "西红柿 180g" → Use 180g
- "鸡蛋 1个 (约50g)" → Use 50g

**Ranges:**
- "盐 1.5-2g" → Use average: 1.75g
- "糖 0-2g" → Use average: 1g

**Volumetric Measures:**
- "植物油 15g" (already in grams) → Use 15g
- "植物油 15ml" → Convert using density: 15ml × 0.92g/ml = 13.8g
- "蒸鱼豉油 10g" → Use 10g

**Estimated Consumption:**
- Frying oil: Recipe uses 500ml, assume ~30ml (27.6g) is absorbed.
- Coating starch: Recipe uses 100g, assume ~60g adheres to food.

### 3. Missing Ingredients
If an ingredient is not in the nutrition database:
1. Check for close alternatives (e.g., "鲜香菇" → "蟹味菇")
2. If negligible (garnishes, spices < 5g), skip.
3. If significant, make a reasonable estimate based on similar ingredients.

## Standard Conversions
- 1个鸡蛋 ≈ 50g
- 1个西红柿 ≈ 180g
- 1根胡萝卜 ≈ 150g
- 1个中等苹果 ≈ 200g
- 1根中等香蕉 ≈ 120g
- 植物油: 1ml ≈ 0.92g
- 酸奶/牛奶: 1ml ≈ 1.03-1.05g
- 蒸鱼豉油: 1ml ≈ 1.1g
- 香醋: 1ml ≈ 1.05g
