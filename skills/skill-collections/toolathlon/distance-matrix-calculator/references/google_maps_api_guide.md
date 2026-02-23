# Google Maps API Usage Guide

## Core Tools for Distance Matrix Skill

### 1. `google_map-maps_geocode`
Converts a human-readable address into geographic coordinates.
- **Use Case:** Getting precise lat/lng for a known address (e.g., "University of Pennsylvania Bookstore").
- **Input:** `{"address": "Full address string"}`
- **Output:** Contains `location.lat`, `location.lng`, `formatted_address`.

### 2. `google_map-maps_search_places`
Finds places by name/query near a location. More flexible than geocode for landmarks.
- **Use Case:** Finding "Benjamin Franklin Statue University of Pennsylvania".
- **Input:** `{"query": "Search string", "location": {"latitude": XX.X, "longitude": XX.X}, "radius": 2000}`
- **Output:** Array of `places` with `name`, `formatted_address`, `location`.

### 3. `google_map-maps_distance_matrix`
**THE CORE TOOL.** Calculates travel distance and time between multiple origins and destinations.
- **Use Case:** Generating the complete cost matrix for TSP.
- **Input:**
  