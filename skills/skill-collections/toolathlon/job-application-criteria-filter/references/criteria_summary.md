# Filter Criteria Summary

## Mandatory Criteria (ALL must be met)
1.  **Status:** Entry must have status "Checking".
2.  **Salary:** Minimum salary > $3000.
    *   *Parsing Logic:* Extract the first number from the "Salary Range" field. Ignore currency symbols and text like "/mo".
3.  **Work Type:** Must be "On-site" (as indicated in the "Flexibility" field).
4.  **Position:** Must be "software engineer" OR "software manager" (case-insensitive match).
5.  **Location:** Must be within 500 km of UCLA (University of California, Los Angeles).
    *   *Reference Coordinates:* UCLA is approximately at lat 34.0699182, lng -118.4438495.
    *   *Logical Distance Check:* Major California cities (LA, San Diego, Long Beach) are within range. International cities are not.

## Action Trigger
When all five criteria above are satisfied for a database entry:
1.  Send an application email using the template.
2.  Update the entry's "Status" to "Applied".
