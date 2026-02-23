# Common Mistakes in Multi-Object Collection Tasks

## Analysis from Trajectory Data

### 1. Redundant Searching
**Mistake:** Checking the same empty container multiple times consecutively
**Example from trajectory:** After finding cellphones on desk, agent rechecked all drawers
**Solution:** Maintain search history and skip recently-searched empty locations

### 2. Delayed Placement
**Mistake:** Continuing search after finding first object instead of placing it immediately
**Optimal pattern:** Find → Place → Continue search
**Rationale:** Reduces risk of losing track or dropping object

### 3. Inefficient Search Order
**Mistake:** Searching all containers before checking visible surfaces
**Observation:** Cellphones were on desk (visible), but agent searched 5 drawers first
**Optimal order:** Visible surfaces → Closed containers

### 4. State Management Errors
**Mistake:** Not closing containers after checking
**Impact:** Environment clutter, potential for navigation issues
**Best practice:** Always close containers unless actively retrieving object

### 5. Quantity Confusion
**Mistake:** Losing count of collected vs. needed items
**Solution:** Explicit tracking with counters updated on each placement

## Efficiency Patterns Identified

### High-Efficiency Pattern:
