---
name: public-transit-route-planner
description: When the user requests public transportation directions between two locations with specific constraints (e.g., shortest walking distance, preferred transport modes, or avoiding certain transfers). This skill geocodes addresses, finds nearest transit stations, calculates optimal routes using transit APIs, identifies transfer points, compiles sequential station lists, and provides detailed journey summaries including distances and travel times.
---
# Instructions

## Primary Objective
Plan an optimal public transit route between two locations based on user constraints (e.g., shortest walking distance, preferred modes, avoiding certain transfers). The final output must include a detailed journey summary and a sequential list of all stations passed, saved to a specified file.

## Core Workflow

### 1. Interpret Request & Extract Constraints
- Identify the **origin** and **destination** addresses.
- Identify explicit **constraints** (e.g., "prefer not to use any mode other than MRT", "shortest walking distance").
- Identify any **output requirements** (e.g., save station list to a file named `routine.txt`).

### 2. Geocode Locations
- Use `google_map-maps_geocode` to obtain precise coordinates and formatted addresses for both origin and destination.
- Store the results for subsequent steps.

### 3. Find Nearest Transit Stations
- Use `google_map-maps_search_places` with query "MRT station" (or other relevant transit types) near the geocoded origin.
- Use `google_map-maps_distance_matrix` (mode: "walking") to compare walking distances to the nearest candidate stations. **Select the station with the shortest walking distance** to honor the "shortest walking distance" constraint.

### 4. Calculate Optimal Transit Route
- Use `google_map-maps_directions` with mode: "transit" between the origin and destination addresses.
- Analyze the returned route steps. The route will typically include:
    - A walking leg to the first station.
    - One or more transit (e.g., "SUBWAY") legs.
    - Possible transfers (indicated by a walking leg between transit legs).
    - A final walking leg to the destination.
- **Validate the route against user constraints.** If the user specifies "MRT only," ensure the transit legs use the subway system.

### 5. Enrich Route Details & Compile Station List
- The `directions` API provides a high-level route but may not list every intermediate station.
- To compile a **complete, sequential list of all stations passed**:
    - Identify each transit leg's line and direction (e.g., "Downtown Line towards Expo").
    - Use `local-web_search` or `fetch-fetch_markdown` to find authoritative lists of stations for each line segment (e.g., Wikipedia pages for "Downtown Line" and "East–West Line").
    - Map the route's transfer points to these lists to extract the exact sequence of stations between the boarding station and the alighting/transfer station for each leg.
- The final list must be **sequential**, including the origin station, all intermediate stations, transfer stations, and the destination station.

### 6. Generate Output & Save File
- Create a final summary for the user including:
    - Total journey distance and time.
    - Walking distance to the first station.
    - Step-by-step itinerary (walk, take Line X from Station A to Station B, transfer, etc.).
    - The complete list of stations in order.
- Use `filesystem-write_file` to save **only the station names, one per line**, to the user-specified file (e.g., `routine.txt`). Do not include any other text in this file.

### 7. Claim Completion
- Use `local-claim_done` to signal successful task completion after providing the summary and confirming the file has been saved.

## Key Decision Points & Error Handling
- **Nearest Station Selection:** If multiple stations are equally close, choose the one that leads to a simpler route (fewer transfers).
- **Data Enrichment:** If web searches for station lists fail or return ambiguous data, rely on the station names explicitly mentioned in the `directions` result and note any potential gaps in the output summary.
- **Constraint Violation:** If the optimal transit route violates a user constraint (e.g., uses a bus), inform the user and explain the limitation, or search for an alternative if possible.
- **File Operations:** If the specified file path is invalid or unwritable, inform the user and ask for an alternative before proceeding.

## Tools Usage Summary
1.  `google_map-maps_geocode`: Convert addresses to coordinates.
2.  `google_map-maps_search_places`: Find transit stations near a point.
3.  `google_map-maps_distance_matrix`: Compare walking distances to stations.
4.  `google_map-maps_directions`: Get the primary transit route.
5.  `local-web_search` / `fetch-fetch_markdown`: Find detailed station lists for specific transit lines.
6.  `filesystem-write_file`: Save the final station list.
7.  `local-claim_done`: Signal task completion.
