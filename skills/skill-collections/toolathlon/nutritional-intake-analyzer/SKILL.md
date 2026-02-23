---
name: nutritional-intake-analyzer
description: Analyzes dietary intake against fitness/nutritional guidelines for fitness planning, fat loss, or muscle gain goals. Reads body metrics, extracts meal ingredients, retrieves nutritional data, calculates expected intake based on guidelines, compares actual vs expected values, and generates formatted assessment reports.
---
# Instructions

## Core Workflow
When the user requests a nutritional analysis of their meals against fitness guidelines, follow this exact sequence:

1.  **Read Input Files:** Use `filesystem-read_multiple_files` to read:
    *   `body.md` (contains height, weight, gender)
    *   `cuisine.md` or similar (contains meal list/recipe names)
    *   `health_guide.md` or similar (contains fitness dietary ratios)
    *   `format.md` (contains the exact output format specification)

2.  **Read Nutritional Database:** Use `excel-get_workbook_metadata` and `excel-read_data_from_excel` to load the nutritional composition spreadsheet (e.g., `Nutrition.xlsx`). Parse it into a lookup dictionary mapping ingredient names to their per-100g nutrient values (focus on carbohydrates and protein).

3.  **Get Recipe Details:** For each dish name listed in the meal file, use `howtocook-mcp_howtocook_getRecipeById` to fetch the detailed recipe, including ingredient lists and quantities.

4.  **Calculate Expected Intake:**
    *   Parse the user's body metrics (height, weight, gender, goal phase).
    *   Calculate BMI: `weight_kg / (height_m ** 2)`.
    *   Consult the fitness guidelines (`health_guide.md`) to determine the appropriate `g/kg` ratios for carbohydrates and protein based on:
        *   Goal (Muscle Gain / Fat Loss)
        *   Phase (Initial / Late)
        *   Gender
        *   Training vs. Rest Day (note: rest day carb intake is typically 0.5 g/kg lower)
        *   BMI adjustments (for large-bodied obese individuals, use reduced ratios).
    *   Calculate expected daily intake ranges/targets:
        *   Carbs (g): `weight_kg * carb_g_per_kg_range`
        *   Protein (g): `weight_kg * protein_g_per_kg_target`

5.  **Calculate Actual Intake:**
    *   For each recipe, map its ingredients and estimated consumed quantities to the nutritional database.
    *   **Handle estimation:** When ingredient quantities are given as ranges (e.g., "1.5-2g"), use the average. For cooking oils, estimate absorbed amounts (e.g., 30ml from 500ml used for frying). For coatings like starch, estimate the portion that adheres to the food.
    *   Sum the carbohydrates and protein from all meals to get the total daily actual intake.

6.  **Perform Assessment:** Compare actual vs. expected values using the assessment logic defined in `format.md`:
    *   **For a range (Carbohydrates):**
        *   `Below expectations`: Actual < 95% of the range's lower bound.
        *   `Excessive intake`: Actual > 105% of the range's upper bound.
        *   `Meets expectations`: Otherwise.
    *   **For a target with tolerance (Protein):** Apply the ±10g tolerance to create an effective range, then use the same 95%/105% rule.

7.  **Generate Output:** Create a new file (e.g., `analysis.md`). Write the analysis using the **exact format** specified in `format.md`, including the required heading, assessment labels, expected values, actual values, and the "|" separator.

8.  **Verify and Conclude:** Read back the generated file to confirm correctness, then provide a final summary to the user.

## Critical Notes
*   **Strict Format Adherence:** The output in `analysis.md` must match the structure in `format.md` character-for-character, including the heading `# Today's Meal Nutritional Analysis`.
*   **BMI-Based Adjustments:** Always check BMI and apply the special reduced intake ratios for "large-bodied obese individuals" (BMI > 28 or > 32) as specified in the guidelines.
*   **Ingredient Mapping:** Be prepared for ingredient name mismatches between the recipe and nutrition database. Use partial matching or manual lookup if necessary.
*   **Quantity Estimation:** Explicitly state your estimation logic when precise consumption amounts are not clear from the recipe (e.g., "assumed 30ml of oil was absorbed").
