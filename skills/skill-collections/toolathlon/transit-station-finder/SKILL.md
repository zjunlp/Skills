---
name: transit-station-finder
description: When the user needs to identify nearby public transit stations relative to a specific location. This skill searches for transit stations (MRT, subway, bus, etc.) within a specified radius of given coordinates, returning station names, distances, and accessibility information to determine optimal boarding points.
---
# Instructions

## Primary Objective
Find the nearest public transit station(s) to a given location and provide detailed information to help the user choose the optimal boarding point for their journey.

## Core Workflow

### 1. Location Identification
- **User provides**: A location (address, landmark, or coordinates).
- **Your action**: Use `google_map-maps_geocode` to obtain precise coordinates and formatted address.
- **Output needed**: Latitude, longitude, and verified address.

### 2. Station Discovery
- **Goal**: Find all transit stations near the identified coordinates.
- **Your action**: Use `google_map-maps_search_places` with:
  - Query: "MRT station" or "transit station" (adapt based on local terminology)
  - Location: The coordinates from step 1
  - Radius: Start with 1000 meters (adjust based on context)
- **Output needed**: List of stations with names, addresses, and distances.

### 3. Distance Verification
- **Goal**: Determine exact walking distances to the most promising stations.
- **Your action**: Use `google_map-maps_distance_matrix` with:
  - Origins: The user's location
  - Destinations: Top 3-5 candidate stations from step 2
  - Mode: "walking"
- **Output needed**: Precise walking distances and times for each station.

### 4. Route Planning (When Destination Provided)
- **Goal**: If user provides a destination, find complete transit route.
- **Your action**: Use `google_map-maps_directions` with:
  - Origin: User's location
  - Destination: Their target destination
  - Mode: "transit"
- **Output needed**: Complete route with walking segments, transit segments, and transfer points.

### 5. Station Details Enrichment
- **Goal**: Gather additional information about stations and lines.
- **Your action**: Use `local-web_search` or `fetch-fetch_markdown` to:
  - Find station line information
  - Identify transfer possibilities
  - Get service schedules if needed
- **Output needed**: Line colors, station codes, transfer options.

### 6. Optimal Station Selection
- **Criteria for selection** (in priority order):
  1. Shortest walking distance
  2. Fewest transfers to destination (if destination provided)
  3. Direct line to destination
  4. Station amenities/accessibility

### 7. Output Format
- **Always include**:
  - Nearest station(s) with walking distances
  - Station codes and line information
  - Next steps for the journey
- **When destination provided**:
  - Complete station-by-station route
  - Total journey time
  - Transfer points

## Special Considerations

### File Output Requirements
If the user requests saving station lists to a file:
- Use `filesystem-write_file` to create the file
- Format: One station per line in sequential journey order
- Include both boarding and intermediate stations
- Example filename: `routine.txt`

### Error Handling
- If no stations found within initial radius, expand search radius incrementally
- If geocoding fails, ask for more specific location details
- If walking distance exceeds reasonable limits (e.g., >2km), suggest alternative transport to station

### Local Adaptations
- Adjust search terms based on local transit systems (e.g., "subway", "metro", "bus terminal")
- Consider local naming conventions for station types
- Account for different transit authorities in search results
