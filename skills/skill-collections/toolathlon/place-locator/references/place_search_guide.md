# Place Search Strategy Guide

## Formulating Effective Queries
The `maps_search_places` tool is powerful but requires well-structured queries. Follow this pattern:

**`[Specific Place Name] + [Context/Location]`**

*   **Good:** `"Benjamin Franklin Statue University of Pennsylvania Philadelphia"`
*   **Good:** `"Fisher Fine Arts Library University of Pennsylvania"`
*   **Better:** `"Moore Building 200 S 33rd St Philadelphia"` (Use when you have a known address from context)
*   **Poor:** `"ENIAC"` (Too vague, likely to return irrelevant results)

## Interpreting Search Results
1.  **Check `types`:** Prefer results with specific types like `tourist_attraction`, `museum`, `library`, `point_of_interest` over generic ones like `establishment`.
2.  **Verify Name Match:** The `name` field should closely match the place you're looking for.
3.  **Use `place_id`:** This unique identifier is crucial for subsequent API calls (e.g., getting detailed place info or photos).

## Handling Common Scenarios
*   **Multiple Listings:** If a search for "College Hall" returns many results, look for the one with the address closest to your known area or with the most specific type.
*   **No Direct Results:** If a search for a specific exhibit ("ENIAC") fails, search for the building or institution that houses it ("Moore Building Penn Engineering").
*   **Geocoding as Fallback:** If a place has a clear address in the source material, use `maps_geocode` on that address as a direct method.

## Data Structure for Output
Maintain a consistent data structure for located places. Example format for your internal notes:

