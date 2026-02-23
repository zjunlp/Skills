# Excel Data Structure Reference

## Methodology Sheet Structure
The Methodology sheet defines how business segments are composed from data sources.

### Header Row (Row 2)
- Column A: Typically empty or label
- Column B: "Segment/Source" header
- Columns C-Q: Source names (e.g., Agriculture, Oil & Gas, Food, Metals, Electronic, Electric, Auto Production, Transportation, Furniture, Construction, Industry, Steel Production, Air Bus, Boeing, MRO)

### Segment Rows (Rows 4-13)
- Column A: Typically empty
- Column B: Segment name (e.g., Auto OEM, Auto Components, Coil, Industry, ACE, Appliance, AlFi, CF, Food, Aero)
- Columns C-Q: Weight values (0-1) mapping the segment to each source
- Row 14: "Total" row (should be excluded from calculations)

### Example Mapping
| Segment      | Agriculture | Oil & Gas | ... | Auto Production | ... |
|--------------|-------------|-----------|-----|-----------------|-----|
| Auto OEM     |             |           |     | 1.0             |     |
| Auto Components |           |           |     | 0.9             |     |
| ...          | ...         | ...       | ... | ...             | ... |

## RawData Sheet Structure
The RawData sheet contains time-series data for each source.

### Header Rows (Rows 1-4)
- Row 1: Region label (e.g., "APAC")
- Row 2: Data source names (should match Methodology source names)
- Row 3: Units (e.g., "mn USD", "units")
- Row 4: Source names (should match Methodology source names exactly)

### Data Rows (Rows 5+)
- Column A: Year values (e.g., 2014, 2015, ...)
- Columns B-P: Data values for each source corresponding to the year

### Example Data
| Year | Agriculture | Oil & Gas | ... | Auto Production | ... |
|------|-------------|-----------|-----|-----------------|-----|
| 2014 | 1984154.14  | 380048.54 | ... | (null)          | ... |
| 2015 | 2026880.45  | 391674.25 | ... | (null)          | ... |
| ...  | ...         | ...       | ... | ...             | ... |

## Calculation Rules
1. **Year-over-Year Growth Formula**: `((current_year - previous_year) / previous_year) * 100`
2. **Missing Data Handling**: 
   - If either current or previous year value is null/missing, growth rate is `None`
   - If previous year value is 0, growth rate is `None` (to avoid division by zero)
   - Formula cells (text starting with '=') are treated as null
3. **Segment Growth Calculation**:
   - For each segment, retrieve its weight mapping from Methodology
   - For each year, check if all component sources have valid growth rates
   - If any source growth is `None`, segment growth is "N/A"
   - Otherwise, calculate weighted sum: `sum(weight * source_growth)`
   - Round result to one decimal place
4. **Output Format**:
   - First column: "Year"
   - Subsequent columns: Segment names in Methodology sheet order (excluding "Total")
   - Values: Growth rate percentages (rounded to 1 decimal) or "N/A"
