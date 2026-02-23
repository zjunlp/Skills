# Troubleshooting Guide

## Common Failure Modes

### 1. "Nothing happened" after clean action
**Possible Causes:**
- You are not at the correct receptacle. Ensure you have executed `go to {recep}` first.
- The object is not in your inventory. You must `take` the object before cleaning.
- The receptacle is not valid for cleaning (e.g., `cabinet`). Use only `sinkbasin`.

**Solution:**
1. Verify your location with `Observation`.
2. Verify inventory by checking if you recently picked up the object.
3. Ensure the receptacle name matches exactly (e.g., `sinkbasin 1`).

### 2. Cannot find a cleaning receptacle
**Solution:**
- Scan the initial observation for `sinkbasin`. There is typically at least one.
- If no sink is present, the task may not require cleaning or the environment is anomalous.

### 3. Object not found in inventory
**Solution:**
- You must acquire the object first (e.g., from a `fridge`, `cabinet`).
- Execute `take {obj} from {recep}` before attempting to clean.

## Success Verification
A successful clean action will yield an observation similar to:
- "You clean the {object_name} using the {receptacle_name}."
- This confirms the object's state is now "clean" for subsequent task steps.
