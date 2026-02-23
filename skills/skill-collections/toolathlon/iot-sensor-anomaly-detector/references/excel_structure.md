# Excel Configuration File Structure

## Expected Sheet: "Operating Parameters"

### Column Structure
| Column | Description | Data Type | Example |
|--------|-------------|-----------|---------|
| A: machine_id | Machine identifier | String | M001, M002 |
| B: machine_description | Machine description | String | "Assembly Line A - Component Insertion" |
| C: sensor_type | Type of sensor | String | temperature, pressure, rpm |
| D: unit | Measurement unit | String | °C, bar, rpm |
| E: min_value | Minimum acceptable value | Numeric | 18, 0.8, 1200 |
| F: max_value | Maximum acceptable value | Numeric | 25, 1.2, 1800 |
| G: calibration_date | Last calibration date | Date | 2024-01-15 |
| H: next_maintenance | Next maintenance date | Date | 2024-07-15 |

### Key Points
1. **Sheet name**: The primary sheet is typically named "Operating Parameters" but may vary.
2. **Header row**: First row contains column headers.
3. **Data types**: Ensure min_value and max_value are numeric values.
4. **Completeness**: All machines and sensor types should have corresponding parameter ranges.

## Alternative Sheet Names
If "Operating Parameters" sheet is not found, check for:
- Parameters
- Operating Ranges
- Machine Parameters
- Sensor Limits
- Config

## Validation Rules
1. `min_value` should be less than `max_value` for each row.
2. `machine_id` and `sensor_type` combinations should be unique.
3. No NULL values in `machine_id`, `sensor_type`, `min_value`, or `max_value`.

## Common Issues
1. **Sheet not found**: Use workbook metadata to discover available sheets.
2. **Column order variation**: The script uses column indices; verify if structure differs.
3. **Data type mismatches**: Ensure numeric columns contain actual numbers, not strings.
