# ALFWorld Task Grammar Reference

This document outlines common patterns in ALFWorld task descriptions to aid in parsing.

## Core Task Structure
Most tasks follow this pattern:
`[Action Phrase] [Object Phrase] and [Placement Phrase] [Receptacle Phrase]`

### 1. Action Phrase
- `find` (e.g., "find two pen")
- `put` (e.g., "put the mug")
- `take` (e.g., "take a book")
- `clean` (e.g., "clean the plate")
- `heat`/`cool` (e.g., "heat the potato")

### 2. Object Phrase
- **Quantity + Object:** `two pen`, `a mug`, `an apple`, `the book`
- **With Identifier:** `pen 1`, `mug 2`, `apple 3`
- **Compound Objects:** `two pen and a pencil` (multiple objects)

### 3. Placement Phrase
- `put in/on`
- `place in/on`
- `put them in/on` (when referring to previously mentioned objects)

### 4. Receptacle Phrase
- `garbagecan`
- `sidetable`
- `drawer`
- `diningtable`
- Often includes identifier: `garbagecan 1`, `sidetable 2`

## Common Task Examples
1. **Simple Placement:** "put the mug on the sidetable"
   - Objects: ["mug"]
   - Receptacle: "sidetable"

2. **Find and Place Multiple:** "find two pen and put them in garbagecan"
   - Objects: ["pen"] (quantity: 2)
   - Receptacle: "garbagecan"

3. **Compound Objects:** "find a pen and a pencil and put them in drawer 1"
   - Objects: ["pen", "pencil"]
   - Receptacle: "drawer 1"

4. **Action with Tool:** "clean the plate with soap"
   - Objects: ["plate"]
   - Tool: "soap" (not a receptacle for placement)

## Parsing Guidelines
- **Object Identification:** The base object name (e.g., "pen") is more important than the instance number ("pen 3") for goal tracking. The goal is usually satisfied by any instance of the object type.
- **Receptacle Context:** The observation text must be analyzed in conjunction with the agent's location. An object listed in an observation is typically in the receptacle the agent is currently at or interacting with.
- **Success Condition:** All target object types must be present in the target receptacle. The specific instance IDs (e.g., "pen 2" vs "pen 3") do not matter unless specified in the goal (rare).
