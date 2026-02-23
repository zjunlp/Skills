# Nutritional Assessment Logic

## Input Requirements
1. **Body Metrics**: Height (cm), Weight (kg), Gender, Goal (Muscle Gain/Fat Loss), Phase (Initial/Late)
2. **Fitness Guidelines**: Document specifying `g/kg` intake ratios for different scenarios.
3. **Meal List**: List of dishes consumed.
4. **Recipe Details**: Ingredient lists with quantities for each dish.
5. **Nutrition Database**: Per-100g nutrient values for ingredients.

## Calculation Steps

### Step 1: Determine User Context
1. Calculate BMI: `weight_kg / (height_meters^2)`
2. Identify:
   - Goal: Muscle Gain / Fat Loss
   - Phase: Initial / Late
   - Day Type: Training / Rest
   - Obesity Category: BMI > 28, BMI > 32, or Normal

### Step 2: Look Up Target Ratios
Consult the fitness guidelines table. Example structure:

| Goal       | Gender | Phase   | Carbs (g/kg) | Protein (g/kg) | Notes                          |
|------------|--------|---------|--------------|----------------|--------------------------------|
| Fat Loss   | Male   | Initial | 2.5-3.0      | 1.5            | Rest day: reduce carbs by 0.5g/kg |
| Fat Loss   | Male   | Initial | 2.4-2.7      | 1.2            | **If BMI > 28**                |
| Fat Loss   | Male   | Initial | 2.0-2.4      | 1.0            | **If BMI > 32**                |

### Step 3: Calculate Expected Intake
- Expected Carbs (g) = `weight_kg × carb_g_per_kg_range`
- Expected Protein (g) = `weight_kg × protein_g_per_kg_target`

### Step 4: Calculate Actual Intake
For each dish:
1. Map each ingredient to the nutrition database.
2. Convert quantity to grams.
3. Calculate nutrient contribution: `(quantity_g / 100) × nutrient_per_100g`
4. Sum across all dishes.

**Important Estimation Rules:**
- For quantity ranges: Use the average.
- For cooking oil: Estimate absorbed amount (e.g., 30ml from 500ml frying oil).
- For coatings (starch, flour): Estimate adhered portion.
- Ignore trace ingredients (garnishes, spices) if nutritional data is unavailable.

### Step 5: Perform Assessment

#### For Carbohydrates (Range)
- **Below expectations**: `actual < 0.95 × expected_min`
- **Excessive intake**: `actual > 1.05 × expected_max`
- **Meets expectations**: Otherwise

#### For Protein (Target with Tolerance)
1. Apply tolerance: Effective range = `target ± tolerance` (typically ±10g)
2. **Below expectations**: `actual < 0.95 × (target - tolerance)`
3. **Excessive intake**: `actual > 1.05 × (target + tolerance)`
4. **Meets expectations**: Otherwise

## Output Format
Strictly follow the template:
