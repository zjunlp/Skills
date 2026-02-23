---
name: multi-traveler-train-coordination
description: Coordinates train travel for multiple people with synchronization requirements (arriving/departing from different locations but meeting at a common destination with time constraints).
---
# Instructions

## 1. Parse User Request
- Identify all travelers, their departure/return cities, and stations.
- Extract the destination city and the requirement to arrive at/depart from the same station there.
- Determine the travel dates (outbound and return) based on relative descriptions (e.g., "next Thursday", "Sunday afternoon") and the **current date**. Use `rail_12306-get-current-date` to get the reference date.
- Note all time constraints:
    - Outbound: Earliest departure time (e.g., "after 5 PM").
    - Return: Time window for departure (e.g., "between 2 PM and 6 PM").
    - Synchronization: Maximum allowed difference in arrival times at the destination and in departure times from the destination (e.g., "should not exceed 30 minutes").
- Confirm the required train types (e.g., "direct high-speed trains (高铁) or EMU trains (动车)").
- The goal is to find **one** valid combination per requested day. Ignore ticket availability. If no combination satisfies all constraints for a day, return `null` for that day.

## 2. Load the Required Output Format
- Read the format specification from the provided `format.json` file using `filesystem-read_file`. The expected structure is typically a JSON object with keys for each travel day (e.g., `"thursday"`, `"sunday"`), each containing sub-objects for each traveler's route.

## 3. Resolve Station Information
- For each mentioned station (departure and destination), obtain its official station code.
    - Use `rail_12306-get-station-code-by-names` for specific station names (e.g., "北京南", "上海虹桥").
    - For the destination city, use `rail_12306-get-stations-code-in-city` to get codes for all stations in that city (e.g., "曲阜"). This is crucial for finding trains to/from the *same* city station.
- **Critical**: Use the **official English translations** for station names in the final output (e.g., "Beijing Nan", "Shanghai Hongqiao", "Qufu East").

## 4. Search for Available Trains
- For the **outbound** date, search for trains from each traveler's departure station to **each** of the destination city's stations.
    - Apply the `earliestStartTime` filter.
    - Use `trainFilterFlags: "GD"` to filter for high-speed (G) or bullet (D) trains as required.
- For the **return** date, search for trains from **each** of the destination city's stations to each traveler's return station.
    - Apply both `earliestStartTime` and `latestStartTime` filters for the specified window.

## 5. Find Synchronized Combinations
- **Arrival Synchronization (Outbound)**:
    1. Group search results by the *destination station* in the target city.
    2. For each destination station, find pairs of trains (one for each traveler) where the absolute difference in their arrival times is ≤ the allowed limit (e.g., 30 minutes).
- **Departure Synchronization (Return)**:
    1. Group search results by the *departure station* in the target city.
    2. For each departure station, find pairs of trains (one for each traveler) where the absolute difference in their departure times is ≤ the allowed limit.
- The arrival and departure stations in the destination city do **not** have to be the same.
- Select **one** valid pair for the outbound journey and **one** valid pair for the return journey. If multiple exist, choose any.

## 6. Format and Output Results
- Construct a JSON object strictly adhering to the loaded `format.json` schema.
- Populate it with the selected train details: train number, departure station, arrival station, departure time, arrival time. **Use official English station names**.
- If no valid combination is found for a day, set its value to `null`.
- Write the final JSON object to the specified output file (e.g., `train-ticket-plan.json`) using `filesystem-write_file`.
- Optionally, read the file back to verify and provide a summary to the user.
- Conclude the task with `local-claim_done`.

## Key Triggers & Handling
- This skill activates when a user request involves coordinating travel for **multiple people** with **synchronization constraints** (time differences, same station) to/from a common destination.
- Be prepared to handle API errors (e.g., "get cookie failed", "City not found") with retries or alternative queries.
- The core challenge is the combinatorial matching under time constraints, not checking seat availability.
