# Threshold Evaluator - Usage Examples

## Example 1: Temperature Measurement (from Trajectory)

**Instruction:**
> "If the melting point of chocolate is above 50.0 degrees, focus on the green box. If the melting point of chocolate is below 50.0 degrees, focus on the blue box."

**Observation:**
> "the thermometer measures a temperature of 56 degrees celsius"

**Skill Execution:**
1. **Extract Measurement:** `56.0`
2. **Parse Condition:** 
   - Threshold: `50.0`
   - Operator: `>` (from "above 50.0 degrees")
   - Action if True: focus on green box
   - Action if False: focus on blue box
3. **Evaluate:** `56.0 > 50.0` = `True`
4. **Execute Branch:** `focus on green box`

**Common Mistake to Avoid:**
- Incorrect: Executing `focus on blue box` when value is above threshold.

## Example 2: Weight Comparison

**Instruction:**
> "If the sample weighs less than 100 grams, use the small container. If it weighs more than 100 grams, use the large container."

**Observation:**
> "The scale reads 87.5 grams"

**Skill Execution:**
1. **Extract Measurement:** `87.5`
2. **Parse Condition:**
   - Threshold: `100.0`
   - Operator: `<` (from "less than 100 grams")
   - Action if True: use small container
   - Action if False: use large container
3. **Evaluate:** `87.5 < 100.0` = `True`
4. **Execute Branch:** `use small container`

## Example 3: pH Level Check

**Instruction:**
> "When the pH drops below 7.0, add the alkaline solution."

**Observation:**
> "pH meter shows 6.8"

**Skill Execution:**
1. **Extract Measurement:** `6.8`
2. **Parse Condition:**
   - Threshold: `7.0`
   - Operator: `<` (from "below 7.0")
   - Action if True: add alkaline solution
   - Action if False: (none specified - continue monitoring)
3. **Evaluate:** `6.8 < 7.0` = `True`
4. **Execute Branch:** `add alkaline solution`

## Edge Cases & Notes

1. **Exact Threshold Match:**
   - Instruction: "above 50 degrees" means `measurement > 50`, not `>= 50`.
   - If measurement equals exactly 50.0, the condition is `False`.
   - Always follow the precise wording of the instruction.

2. **Multiple Measurements:**
   - If taking multiple readings (e.g., monitoring temperature rise), evaluate after each measurement.
   - The skill should be triggered for each new measurement observation.

3. **Implicit Thresholds:**
   - Sometimes the threshold is implied: "If it's too hot (>40°C), turn off the heater."
   - Extract the numerical boundary from context.

4. **Unit Consistency:**
   - Ensure the measurement and threshold are in the same units.
   - The extraction function handles unit-agnostic number parsing.

## Integration Pattern

