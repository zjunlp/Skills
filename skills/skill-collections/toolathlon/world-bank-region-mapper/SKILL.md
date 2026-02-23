---
name: world-bank-region-mapper
description: When the user needs to classify countries according to World Bank geographical or income-based classifications for economic analysis. This skill provides accurate mappings between country names and World Bank region codes (EAS, ECS, LCN, MEA, NAC, SAS, SSF) and income groups (LIC, LMC, UMC, HIC), handling variations in country naming conventions. Triggers include 'World Bank classification', 'map countries to regions', 'regional grouping', 'country classification', 'geographical regions', and any analysis requiring standardized regional categorization according to World Bank definitions.
---
# Instructions

## Primary Use Case: CR5 Index Calculation
This skill is optimized for calculating economic concentration indices (like CR5) across World Bank geographical regions. The canonical workflow is:
1.  **Locate Source Data:** Identify a Google Sheets spreadsheet containing country-level GDP/economic data and a separate sheet or source with regional GDP totals.
2.  **Extract and Map:** Load the country data, map each country to its correct World Bank geographical region using the provided `country_to_region_mapping.json`.
3.  **Group and Calculate:** For each of the 7 major regions (EAS, ECS, LCN, MEA, NAC, SAS, SSF):
    *   Filter and sort the countries within the region by economic value (e.g., GDP) in descending order.
    *   Identify the top N countries (default is top 5, or fewer if the region has fewer countries).
    *   Calculate the sum of the top N countries' values.
    *   Retrieve the region's total value from the regional data source.
    *   Compute the concentration ratio: `(Top N Sum / Region Total) * 100`. Round to two decimal places.
4.  **Output Results:** Create a new, clearly named spreadsheet or table. Populate it with headers: `Region`, `TopN_Countries`, `TopN_Sum`, `Region_Total`, `CRN_Ratio`. Sort the final table in descending order by the concentration ratio.

## Core Mapping Logic
*   **Use the Bundled Mapping:** Always use the canonical `country_to_region_mapping.json` file for the most accurate and up-to-date World Bank classifications. It handles common naming variations found in datasets.
*   **Region Focus:** The seven major geographical regions are the primary focus. Income group classifications (LIC, LMC, UMC, HIC) are available in the mapping but are secondary.
*   **Handling Unmapped Entities:** If a country/entity name from the source data is not found in the mapping, log it for review. Do not guess its classification.

## Key Considerations
*   **Data Integrity:** Verify that the sum of individual country GDPs within a region approximates the official regional total from the World Bank source. Discrepancies may occur due to data vintage or methodology.
*   **Flexible N:** The "N" in CRN (e.g., CR5) is a parameter. The skill should adapt if a region has fewer than N members (e.g., North America has only 3 countries, so CR5 becomes CR3 for that region).
*   **Output Formatting:** Present monetary values in a readable format (e.g., with thousand separators). Express ratios as percentages.

## When to Use Scripts vs. Reasoning
*   **Use `scripts/classify_and_calculate.py` (Low Freedom):** For the deterministic, error-prone tasks of loading data, applying the mapping, performing grouping, sorting, and arithmetic calculations. This ensures consistency and accuracy.
*   **Use Your Judgment (High Freedom):** For interpreting user intent, identifying the correct source sheets in a workbook, designing the structure and naming of the output spreadsheet, and communicating findings or data discrepancies to the user.

## Quick Start Command
For a standard CR5 analysis from Google Sheets, the expected user command pattern is:
"Read the country GDP data from [Source Spreadsheet], calculate the CR5 index for each World Bank region, and save the results to a new spreadsheet named [Output Name] with a table titled [Table Name]."
