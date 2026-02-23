# Troubleshooting: Failed Pickup Actions

If the `take` action results in "Nothing happened," consider the following common issues:

1.  **Agent Position:** The agent must be at the same location as the source receptacle. Verify your location using the last observation.
2.  **Object Visibility:** The object must be visible on the receptacle. Check the last observation for a list of items on the `source_receptacle`.
3.  **Object Name:** Ensure the `object` identifier matches exactly the name shown in the observation (e.g., "toiletpaper 1" not "toiletpaper").
4.  **Receptacle State:** Some receptacles (like cabinets) may need to be opened before their contents can be accessed.
5.  **Inventory Limit:** The agent may have a limited carrying capacity. Ensure you are not already holding the maximum number of items.
