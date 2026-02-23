---
name: geolocation-distance-filter
description: When the user needs to filter locations based on proximity to a reference point within a specified radius. This skill geocodes addresses to coordinates, calculates distances between locations, and filters results based on maximum distance thresholds. Triggers include 'within X km/miles of', 'distance from', 'near location', 'proximity filter', or when location-based filtering is required for job applications, real estate, or service area analysis.
---
# Geolocation Distance Filter

## Purpose
Filter a list of items based on the geographic distance of their locations from a specified reference point. This is essential for location-based queries like finding jobs, properties, or services within a certain radius.

## Core Workflow
1.  **Identify Reference Point:** Determine the central location (e.g., "UCLA", "downtown office") and the maximum allowed distance (e.g., "within 500 km").
2.  **Geocode Reference:** Convert the reference point's address into latitude/longitude coordinates using the `google_map-maps_geocode` tool.
3.  **Extract Target Items:** Obtain the list of items to filter (e.g., from a database query). Each item must have a location field.
4.  **Geocode & Calculate:** For each candidate item:
    *   Geocode its location address.
    *   Calculate the distance from the reference coordinates using the Haversine formula (see `scripts/distance_calculator.py`).
5.  **Apply Filter:** Retain only items where the calculated distance is less than or equal to the specified maximum radius.
6.  **Return Results:** Output the filtered list and the calculated distances for context.

## Key Instructions
*   **Precision:** Always use the `google_map-maps_geocode` tool for reliable coordinate conversion. Do not assume distances.
*   **Unit Consistency:** Ensure the maximum distance threshold and the calculated distances are in the same units (kilometers by default). Convert if necessary (1 mile = 1.60934 km).
*   **Error Handling:** If geocoding fails for an item, log it and exclude it from the filtered results, noting the issue.
*   **Integration:** This filter is typically one step in a larger workflow (e.g., filter jobs, then apply other criteria like salary).

## Common Triggers
*   "Find [items] within [X] km/miles of [location]"
*   "Which [items] are closest to [reference point]?"
*   "Filter these results by distance from [address]"
*   "Show only [items] in the [area] area"
*   Implied need when location is a critical constraint (e.g., "commutable jobs", "nearby stores").

## Bundled Resources
*   `scripts/distance_calculator.py`: Contains the `calculate_distance` function for accurate distance computation.
*   `references/haversine_formula.md`: Explanation of the distance calculation method.
