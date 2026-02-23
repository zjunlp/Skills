---
name: graduate-program-locator
description: When the user needs to find graduate programs based on specific ranking criteria, geographical constraints, and filtering requirements. This skill extracts university rankings from academic ranking websites (like CSRankings), filters programs by specific criteria (ranking position, research areas, year), geocodes locations, calculates driving distances from a reference point, and compiles structured data into formatted reports. It handles web scraping of ranking data, geographic distance calculations, and JSON data formatting for academic program comparisons.
---
# Skill: Graduate Program Locator

## Purpose
Locate and compile a structured list of graduate programs meeting specific ranking, geographical, and filtering criteria for academic planning and application preparation.

## Core Workflow
1.  **Parse User Request:** Identify the key constraints: ranking source, ranking criteria (field, year, top N), geographical reference point, maximum distance, and required output format.
2.  **Extract Ranking Data:** Navigate to the specified ranking website (e.g., CSRankings), configure filters (research areas, year range, country), and extract the ranked list of institutions.
3.  **Geocode Locations:** Obtain precise coordinates for the user-specified reference location and for each qualifying university.
4.  **Calculate Distances:** Compute the driving distance from the reference point to each university.
5.  **Filter & Sort:** Apply the distance constraint, then sort the final list primarily by distance (nearest first), and secondarily by rank (higher rank first) for ties.
6.  **Generate Report:** Format the results into the specified structured output (e.g., JSON) and save the file.

## Key Instructions & Logic

### 1. Configuring the Ranking Source (CSRankings Example)
-   Navigate to the ranking site with parameters pre-set for the desired field (e.g., `https://csrankings.org/#/index?ai&vision&mlmining&nlp&us` for US AI programs).
-   **Critical:** Ensure unwanted sub-fields are deselected. For example, if "Information Retrieval" must be excluded, verify its checkbox is **unchecked**.
-   Set the publication year range precisely. Use the "from year" and "to year" dropdowns to select the same year (e.g., 2024) for a single-year ranking.
-   Allow the page to load completely after each filter change. Listen for console log messages like "Rank took... milliseconds" as an indicator.

### 2. Extracting Ranked Institutions
-   The ranking is presented in a table. Parse rows to capture:
    -   `rank`: The numeric rank (#) from the first column.
    -   `institution_name`: The full university name.
    -   Note: The table may contain tie ranks (e.g., two institutions at rank "3").
-   Extract data for the top N institutions as specified by the user.

### 3. Geocoding & Distance Calculation
-   Use a geocoding tool to get coordinates for the reference address (e.g., "Natural History Museum of Los Angeles County").
-   For each university, construct a search query typically as "University Name, City, State" for reliable results. The city can often be inferred from the university's common location.
-   Use a distance matrix tool to batch-calculate driving distances from the single origin to all university destinations.
-   **Unit Conversion:** The skill typically receives distances in **kilometers**. Convert to miles using the formula: `miles = kilometers * 0.621371`. Round the final result to the nearest integer as requested.

### 4. Data Processing & Output
-   Create a list of candidate universities with fields: `university`, `city`, `cs_ranking_rank`, `car_drive_miles`.
-   Filter out any university where `car_drive_miles > maximum_distance`.
-   Sort the filtered list:
    1.  Primary key: `car_drive_miles` (ascending).
    2.  Secondary key: `cs_ranking_rank` (ascending) for universities with equal distance.
-   Write the sorted list as a properly formatted JSON array to the specified filename.

## Common Pitfalls & Validations
-   **Year Filter:** Double-check that both "from" and "to" year dropdowns are set correctly. An incorrect range will pull data from multiple years.
-   **Field Exclusion:** Visually confirm checkboxes for excluded research areas (e.g., "The Web & information retrieval") are not selected.
-   **Distance Units:** Confirm the distance matrix output is in kilometers before converting to miles.
-   **City Names:** Use the city from the geocoding result (`formatted_address`) for consistency, not assumed locations.
-   **Tie Ranks:** The scraping logic must handle consecutive rows with the same rank number without skipping data.

## Output
The primary deliverable is a JSON file containing the filtered and sorted list of graduate programs. Provide a concise summary to the user highlighting the number of programs found, the closest and highest-ranked options, and the save location of the file.
