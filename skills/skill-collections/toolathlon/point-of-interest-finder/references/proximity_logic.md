# Logic for Determining "Closest" or "Most Convenient"

The user's core constraint is usually proximity. Use this logic to select the best POI from search results.

## Primary Signal: Name and Address Scrutiny
The most reliable indicator is textual analysis of the result's `name` and `formatted_address`.
*   **Example from Trajectory:** User wanted a Starbucks "as close as possible to the Nihonbashi Entrance".
*   **Winning Result:** `"Starbucks Coffee - JR Tokyo Station Nihombashi Entrance"`
*   **Why it won:** The name explicitly contains "Nihombashi Entrance", perfectly matching the user's anchor location. Other results had names like "Gransta Yaesu" or "Tokyo Midtown Yaesu", which are less specific.

## Secondary Signal: Geographic Coordinates
If names are ambiguous, use the `location` data (`lat`, `lng`) of each result.
*   You can use the `google_map-maps_distance_matrix` tool (as seen in the trajectory for a different leg) to compare walking distances from the anchor to each candidate.
*   **Formula for Priority:** Closest straight-line or walking distance wins.

## Tie-Breakers
If proximity is indistinguishable:
1.  **Higher `rating`** may indicate better quality/service.
2.  **Presence of `opening_hours`** and current `open_now` status ensures usability.
3.  User reviews (`reviews`) might mention convenience or accessibility.

## Decision Flowchart
1.  Does any result's name/address **explicitly mention** the user's specified anchor location? → **YES:** Select it.
2.  **NO:** Calculate/estimate distances from the anchor to each result's coordinates. → Select the closest.
3.  **TIE:** Select the one with the higher rating or confirmed open status.
