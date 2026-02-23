# Troubleshooting Guide

## Common Failure Patterns

### 1. "Nothing happened" after `go to {receptacle}`
- **Cause**: The receptacle identifier may be incorrect or the agent is not close enough.
- **Solution**: 
  - Verify the receptacle name from the latest observation.
  - Ensure you are using the exact identifier (e.g., `sidetable 1` not `side table 1`).
  - If multiple receptacles exist, try the closest one first.

### 2. "Nothing happened" after `take {object} from {receptacle}`
- **Cause**: 
  - The object is not on the specified receptacle.
  - The object identifier is incorrect.
  - The object is not takeable.
- **Solution**:
  - Re-examine the observation for the receptacle's contents.
  - Confirm the object exists and is spelled correctly.
  - Try taking a different instance (e.g., `laptop 2` if `laptop 1` fails).

### 3. "Nothing happened" after `put {object} in/on {receptacle}`
- **Cause**:
  - The destination receptacle is full or does not accept the object.
  - The agent is not holding the object.
- **Solution**:
  - Check if the destination has capacity (e.g., not already holding many items).
  - Verify the agent successfully picked up the object in the previous step.
  - Try a different destination receptacle if the task allows.

## Best Practices
1. **Always validate inputs**: Cross-check object and receptacle names with the latest observation.
2. **Order matters**: Execute actions in the sequence order (go → take → go → put).
3. **Handle variants**: Some tasks may require `put {object} in {receptacle}` vs. `on {receptacle}`. Use the preposition that matches the destination type (e.g., `in` for containers, `on` for surfaces).
