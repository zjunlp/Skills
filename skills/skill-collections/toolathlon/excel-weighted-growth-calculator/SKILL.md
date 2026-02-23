---
name: excel-weighted-growth-calculator
description: When the user requests to calculate year-over-year growth rates for business segments using a weighted methodology from Excel files, particularly when there's a mapping sheet defining segment-to-source weights and a raw data sheet with time-series values. This skill handles reading Excel workbooks with multiple sheets, parsing segment/source mappings, calculating YoY growth rates for sources, applying weighted sums to compute segment growth rates, handling missing data with N/A values, and outputting results to a new Excel file with proper formatting.
---
# Instructions

## Overview
This skill calculates year-over-year (YoY) growth rates for business segments based on a weighted methodology defined in an Excel workbook. It reads two key sheets:
1. **Methodology Sheet**: Contains segment-to-source weight mappings.
2. **RawData Sheet**: Contains time-series data for each source.

The skill outputs a new Excel file with segment growth rates for specified years.

## Core Workflow

### 1. Parse Input Requirements
- Identify the input Excel file path (e.g., `Market_Data.xlsx`).
- Determine the output file name (e.g., `segment_growth_rates.xlsx`).
- Confirm the year range for calculations (e.g., 2015-2024).
- Note any special instructions (e.g., exclude "Total" segment, maintain column order).

### 2. Extract Methodology Mappings
- Read the **Methodology** sheet.
- Identify source names from the header row (typically row 2, columns C-Q).
- Identify segment names from column B (rows 4-13, excluding "Total").
- For each segment, extract non-zero weights mapping to sources.
- Preserve the segment order as they appear in the sheet.

### 3. Extract Raw Data
- Read the **RawData** sheet.
- Identify source names from the header row (typically row 4, columns B-P).
- Map each source to its data column.
- Extract year values from column A and map to row numbers.
- For each source and year, read the value. Treat formula cells (strings starting with '=') as `None`/null.

### 4. Calculate Source Growth Rates
- For each source, for each target year (e.g., 2015-2024):
  - Get the current year value and the previous year value.
  - If either value is `None` or previous year is 0, growth rate is `None`.
  - Otherwise, calculate: `((current - previous) / previous) * 100`.
- Store results in a nested dictionary: `source_growth_rates[source][year]`.

### 5. Calculate Segment Growth Rates
- For each segment, for each target year:
  - Retrieve the segment's weight mapping from methodology.
  - For each source in the mapping:
    - Get the source's growth rate for that year.
    - If any source's growth rate is `None`, mark the entire segment's growth as **"N/A"** for that year.
    - Otherwise, compute the weighted sum: `sum(weight * growth_rate)`.
  - Round the result to **one decimal place**.
- Store results: `segment_growth_rates[segment][year]`.

### 6. Generate Output Excel File
- Create a new workbook.
- Write header row: First column "Year", followed by segment names in the preserved order.
- For each target year, write a row: year value followed by each segment's growth rate (number or "N/A").
- Save the file to the specified output path.

### 7. Validate Output
- Verify the file was created successfully.
- Optionally, read back and display a sample to confirm formatting.

## Key Considerations
- **Missing Data**: If a source has no data for current or previous year, its growth is `None`. If any source in a segment's mapping has `None` growth, the segment's growth is "N/A".
- **Formula Cells**: Treat cells with formulas (text starting with '=') as null/unavailable.
- **Segment Order**: Maintain the exact segment order as they appear in the Methodology sheet rows.
- **Output Format**: First column is "Year", subsequent columns are segment names. Growth rates are percentages rounded to one decimal place or "N/A".

## Error Handling
- If the input file or sheets are missing, notify the user.
- If source names between Methodology and RawData sheets don't align, attempt fuzzy matching or notify.
- If year range extends beyond available data, calculate only for available years.

## Usage Example
**User Request**: "Based on the provided file `Market_Data.xlsx`, calculate the year-over-year growth rate percentage for each segment from 2015 to 2024 according to the segment/source mapping provided in the `Methodology` sheet (rounded to one decimal place), and save the results to a new Excel file named `segment_growth_rates.xlsx` where the first column is `Year` and the subsequent columns are the exact names (excluding `Total`) of segments arranged in the same order as they appear in rows in the `Methodology` sheet."

**Skill Execution**:
1. Parse request to identify input file, output name, year range.
2. Execute the workflow above using the bundled script.
3. Deliver the output file.
