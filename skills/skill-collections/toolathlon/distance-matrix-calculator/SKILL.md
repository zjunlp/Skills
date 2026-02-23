---
name: distance-matrix-calculator
description: Calculates distances and travel times between multiple locations for route planning or optimization. Computes a complete distance matrix, supports different travel modes, and returns both distance and time estimates.
---
# Instructions

## Primary Objective
When the user requests route planning, optimization, or distance calculations between multiple points, use this skill to compute the most efficient path. The core task is to generate a **complete distance and time matrix** between all specified locations and then determine the optimal visiting order (solving a Traveling Salesman Problem - TSP) when a starting point is defined.

## Core Workflow

### 1. Parse the Request & Gather Locations
- **Identify Attractions/Destinations:** Extract all locations the user wants to visit from their request and any referenced documents (e.g., `recommendation.md`). List each attraction separately, even if they are close.
- **Identify the Starting Point:** Determine the user's current location or specified starting point (e.g., "I am currently at X").
- **Geocode All Points:** Use `google_map-maps_geocode` or `google_map-maps_search_places` to obtain precise latitude/longitude coordinates and formatted addresses for **every location** (starting point + all destinations).

### 2. Generate the Distance Matrix
- Use `google_map-maps_distance_matrix` with all gathered coordinates as both `origins` and `destinations`.
- Set the `mode` parameter based on the request (e.g., `"walking"`, `"driving"`). Default to `"walking"` if unspecified.
- **Output Analysis:** The tool returns a matrix. Extract the `distance.value` (meters) and `duration.value` (seconds) for every origin-destination pair. Organize this data into clear numerical arrays for calculation.

### 3. Calculate the Optimal Route (TSP)
- **Goal:** Find the shortest total travel path that starts at the given starting point and visits all destination points exactly once.
- **Method:** For a small number of points (n ≤ 6), perform a **brute-force search** over all possible permutations of destinations.
- **Process:**
    1. Fix the starting point as the first location in the sequence.
    2. Generate all possible orders (permutations) for the remaining destination points.
    3. For each permutation, sum the distances between consecutive points using the pre-computed matrix.
    4. Identify the permutation with the **minimum total distance**.
- **Validation:** Manually check a few of the top-performing routes to ensure the result is logical (e.g., doesn't criss-cross unnecessarily).

### 4. Generate Detailed Directions
- For each leg of the chosen optimal route, use `google_map-maps_directions` to get turn-by-turn navigation instructions.
- Extract and format the `summary`, `distance.text`, `duration.text`, and key steps from `steps[n].instructions`.

### 5. Format and Save the Output
- Structure the final plan according to any user-specified format (e.g., a provided `format.json` template).
- The output must include:
    - `road_plan`: An array for each leg with `from`, `to`, `distance`, `estimated_time`, and `directions`.
    - `total_distance`: The sum of all leg distances.
    - `total_time`: The sum of all leg durations.
- Save the final JSON file using `filesystem-write_file`.
- **Verify** the written file using `filesystem-read_file`.

## Key Triggers & User Phrases
- "Plan the shortest walking/driving route"
- "Optimize the visiting order"
- "Calculate distances between these places"
- "Give me a tour of all these attractions"
- "Find the most efficient path"

## Error Handling & Assumptions
- **Missing Coordinates:** If a place cannot be found, use a broader search query or ask the user for clarification.
- **Matrix Errors:** If the distance matrix call fails for some pairs, you may need to call the directions API for those specific pairs as a fallback.
- **Time vs. Distance Optimization:** By default, optimize for **shortest total distance**. If the user specifies "fastest route," optimize for **shortest total time**.
- **Single Destination:** If only one destination is provided, simply return the direct route from start to finish.
