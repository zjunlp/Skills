---
name: scienceworld-container-relocator
description: Moves an object from inventory to a specified container in a target room. Triggered when the task requires placing an item into a particular receptacle (e.g., 'move it to the orange box').
---
# Skill: Container Relocator

## Purpose
This skill handles the final delivery step in scienceWorld experiments where you must place a specific object into a designated container located in a target room.

## Core Logic
1.  **Identify the Target:** The task will specify a destination container (e.g., "the orange box") and a room (e.g., "in the workshop").
2.  **Confirm Inventory:** Ensure the required object is already in your inventory. If not, you must first acquire it.
3.  **Navigate:** Teleport to the target room if you are not already there.
4.  **Execute Delivery:** Use the `move OBJ to OBJ` action to transfer the object from your inventory to the target container.

## Key Instructions
*   Use this skill **only** when the task explicitly states a "move it to [CONTAINER] in [ROOM]" objective.
*   The object to be moved must be in your inventory. Use `focus on OBJ` to confirm the correct target if needed.
*   Always verify your location before attempting the move action. Use `teleport to LOC` to reach the correct room.
*   The primary action is `move [OBJECT] to [CONTAINER]`. Ensure the object name matches your inventory and the container name matches the room's description.

## Example Flow (from trajectory)
1.  Task: "move it to the orange box in the workshop."
2.  Identify: Object = dove egg (in inventory), Container = orange box, Room = workshop.
3.  Action: `teleport to workshop`
4.  Action: `move dove egg to orange box`
