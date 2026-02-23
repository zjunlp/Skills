---
name: point-of-interest-finder
description: When the user needs to locate specific businesses or services near a given location with proximity constraints. This skill handles 1) Searching for businesses by category (e.g., coffee shops, restaurants) within specified radii, 2) Ranking results by proximity to target location, 3) Retrieving detailed business information including names, addresses, ratings, and operating hours, 4) Filtering results based on specific criteria like 'closest to [location]'. Triggers include requests for 'find nearest [business type]', 'locate [service] near [location]', or when users need to incorporate specific stops into their itineraries.
---
# Instructions

Your goal is to find a specific point of interest (POI) for the user, typically to incorporate into a larger plan. Follow this process.

## 1. Clarify the Search Parameters
First, understand the user's exact request. Identify:
*   **Target POI Type:** What is the user looking for? (e.g., "Starbucks", "coffee shop", "pharmacy").
*   **Anchor Location:** The specific place the POI must be near. This is often part of a larger journey (e.g., "near the Nihonbashi Exit of Tokyo Station").
*   **Proximity Constraint:** The user's priority, usually "closest" or "most convenient".
*   **Use Case Context:** How will this POI fit into the user's plan? (e.g., "buy a coffee before departure").

**Do not proceed** until these four parameters are clear, either from the user's request or by asking for clarification.

## 2. Geocode the Anchor Location
Use the `google_map-maps_geocode` tool to get the precise coordinates (`lat`, `lng`) and formatted address of the **Anchor Location**. This is critical for an accurate search.
*   **Input:** The address or name of the anchor location as specified by the user.
*   **Output:** Save the `location` object (containing `lat` and `lng`) for the next step.

## 3. Search for POIs
Use the `google_map-maps_search_places` tool to find candidates.
*   **`query`:** Combine the **Target POI Type** and the **Anchor Location** (e.g., "Starbucks near Tokyo Station Nihonbashi Exit").
*   **`location`:** Use the `location` object obtained in Step 2.
*   **`radius`:** Start with a small radius (e.g., 500 meters). If results are insufficient, you may increase it incrementally.

## 4. Analyze and Rank Results
Examine the returned list of `places`. Your primary ranking factor is **proximity to the Anchor Location**.
1.  Identify the result whose name or address most closely matches the user's **Proximity Constraint** (e.g., "closest to the Nihonbashi Entrance").
2.  If multiple candidates seem equally close, you may consider secondary factors like `rating` or check the `formatted_address` for clearer location hints (e.g., "...Nihombashi Entrance..." in the name).
3.  Select the single best candidate. Its `place_id` is needed for the next step.

## 5. Retrieve Detailed Information
Use the `google_map-maps_place_details` tool with the selected candidate's `place_id`.
*   This provides the definitive, detailed information to present to the user: `name`, `formatted_address`, `formatted_phone_number`, `website`, `rating`, `reviews`, and `opening_hours`.

## 6. Integrate and Present Findings
Present your final recommendation clearly:
1.  **State the recommended POI** by its official `name`.
2.  **Provide its precise `formatted_address`**.
3.  **Justify your choice** based on the user's **Proximity Constraint** (e.g., "This is the Starbucks located at the Nihonbashi entrance, making it the most convenient").
4.  **Include relevant details** like operating hours if they are pertinent to the user's plan.

**Remember:** This skill is a component of a larger task. Your output should be the identified POI information, ready to be slotted into the user's broader plan (e.g., a travel itinerary).
