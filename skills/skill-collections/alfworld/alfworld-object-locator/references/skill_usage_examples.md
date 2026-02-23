# Skill Usage Examples

## Example 1: Basic Location (from trajectory)
**Observation**: "You are in the middle of a room. Looking quickly around you, you see a bathtubbasin 1, a cart 1, a countertop 1, a drawer 4, a drawer 3, a drawer 2, a drawer 1, a garbagecan 1, a handtowelholder 1, a shelf 1, a sinkbasin 1, a toilet 1, a toiletpaperhanger 1, and a towelholder 1."

**Goal**: "put some dishsponge on toilet"

**Agent Thought**: "I need to find the dishsponge. Let me scan the environment for likely locations."

**Skill Output**: "go to cart 1"

**Reasoning**: Dishsponge is mapped to 'cart' in common-sense knowledge, and cart is present in the observation.

## Example 2: Multiple Possible Locations
**Observation**: "You see a countertop 1, a sinkbasin 1, a shelf 2, a shelf 1, and a drawer 1."

**Goal**: "get a spoon"

**Skill Output**: "go to drawer 1"

**Reasoning**: Spoon is strongly associated with drawers in the mappings.

## Example 3: No Clear Match
**Observation**: "You see a toilet 1, a bathtubbasin 1, and a towelholder 1."

**Goal**: "find a plate"

**Skill Output**: None (or "Observation: Target object not found in visible receptacles")

**Reasoning**: Plate typically appears in kitchen areas (countertop, shelf, cart), none of which are present.

## Example 4: Visible Object
**Observation**: "On the countertop 1, you see a plate 1, a cup 1, and a spoon 1."

**Goal**: "take the spoon"

**Skill Not Needed**: The object is already visible in the observation, so no location scanning is required.

## Integration Pattern
