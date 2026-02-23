---
name: traveling-salesman-solver
description: When the user needs to find the optimal visiting order for multiple locations to minimize total travel distance or time, this skill analyzes distance matrices to solve the Traveling Salesman Problem (TSP). It evaluates permutations of locations to identify the shortest route that visits all points. Triggers include 'optimal route', 'shortest path visiting all', 'best visiting order', or when route optimization with multiple stops is required.
---
# Instructions

## Core Workflow
1.  **Identify Locations & Starting Point:** Determine the user's starting location and the list of all destinations that must be visited. Extract these from user request or provided files (e.g., `recommendation.md`).
2.  **Geocode Locations:** Obtain precise coordinates for all locations using mapping tools (`maps_geocode`, `maps_search_places`).
3.  **Build Distance Matrix:** Calculate pairwise walking (or specified mode) distances and times between all locations using `maps_distance_matrix`.
4.  **Solve TSP:** Analyze the distance matrix to find the permutation of destinations that minimizes total travel distance/time from the starting point. For small sets (n ≤ 7), manual permutation analysis is acceptable. For larger sets, use the provided `tsp_solver.py` script.
5.  **Generate Detailed Route:** For each leg of the optimal route, fetch turn-by-turn walking directions using `maps_directions`.
6.  **Format & Deliver Output:** Compile the final route plan into the user's requested format (e.g., JSON matching a template like `format.json`). Save the file and present a clear summary.

## Key Considerations
*   **Clarify Constraints:** Confirm with the user if the route must start/end at specific points, and if visiting time at locations matters (usually it doesn't for pure TSP).
*   **Handle Proximity:** Treat very close locations (e.g., statue and building entrance) as separate stops if the user requires it, even if it slightly increases calculated distance.
*   **Validation:** After writing the output file, read it back to verify its contents and structure.
*   **Assumptions:** The skill assumes travel cost is symmetric and based on network distance/time (walking by default). Real-world constraints like one-way streets are handled by the mapping API.
