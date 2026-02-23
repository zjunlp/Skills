# Common Receptacles in ALFWorld Environment

## Overview
This document lists common receptacle types found in ALFWorld household environments and their typical contents.

## Kitchen Receptacles

### Refrigeration
- **fridge 1**: Cold storage for perishable items
  - Typical contents: vegetables, fruits, dairy, meats
  - Example: potato, tomato, plate, milk

### Cooking Appliances
- **microwave 1**: For heating food
  - May contain: previously heated items
  - Example: apple, mug, plate

### Storage
- **cabinet 1-9**: General storage
  - Contents vary: dishes, food packages, utensils
- **drawer 1-13**: Smaller storage
  - Often contains: utensils, small items
  - Example: fork, spoon, knife

### Countertops
- **countertop 1-2**: Surface areas
  - May hold: various objects temporarily
  - Not typically "opened" but can hold items

## Living/Bedroom Receptacles
- **desk 1**: Work surface with drawers
- **sidetable 1-2**: Small tables with storage
- **dresser 1**: Clothing storage

## Bathroom Receptacles
- **cabinetbath 1**: Bathroom storage
- **drawerbath 1**: Bathroom drawer

## Opening Mechanics

### States
1. **Closed**: Cannot see contents, must open first
2. **Open**: Contents visible, can interact with items
3. **Locked**: Requires key or specific action (rare)

### Action Requirements
- Most receptacles require `open {receptacle}` before accessing contents
- Some may be initially open
- Closing is optional but can be done with `close {receptacle}`

### Special Cases
- **toaster 1**: May require toggling rather than opening
- **coffeemachine 1**: May have specific interaction sequences
- **garbagecan 1**: Usually open, but may have lid

## Best Practices

### Before Opening
1. Verify you are at the receptacle location
2. Check observation for "closed" status
3. Ensure no conflicting actions needed first

### After Opening
1. Note all visible items
2. Check if target item is present
3. Plan next action (take, put, etc.)

### Error Recovery
If "Nothing happened" when trying to open:
1. Re-read observation - receptacle might already be open
2. Check spelling of receptacle identifier
3. Verify you're at correct location
4. Try alternative approach if applicable
