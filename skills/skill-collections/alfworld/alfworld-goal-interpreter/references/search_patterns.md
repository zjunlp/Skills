# ALFWorld Search & Fallback Patterns

## Core Principle
When the primary search path fails, systematically explore the environment using known object-location associations.

## 1. Object Location Priors
Use these common associations to guide search when an object is not in its expected location.

| Object Type      | Common Receptacles/Locations                         | Priority |
|------------------|------------------------------------------------------|----------|
| desklamp         | desk, sidetable, shelf                               | High     |
| pillow           | bed, sofa, armchair                                  | High     |
| book             | desk, shelf, sidetable, bed                          | Medium   |
| cellphone        | sidetable, desk, shelf                               | Medium   |
| keychain         | drawer, sidetable, desk                              | Low      |
| CD               | drawer, shelf, desk                                  | Low      |
| pen/pencil       | desk, drawer, shelf                                  | Medium   |

## 2. Spatial Relation Resolution
For each spatial relation, define the required inspection action.

| Relation | Required Agent Actions (Sequence)                    | Notes                                                                 |
|----------|------------------------------------------------------|-----------------------------------------------------------------------|
| under    | 1. `go to` reference_object. 2. `look`/`examine` area beneath it. | The target may be physically under or logically associated (e.g., in a drawer below). |
| on       | 1. `go to` reference_object. 2. Inspect its surface. | Check for clutter. The target might be obscured by other objects.     |
| in       | 1. `go to` reference_object. 2. `open` it (if closed). 3. Inspect contents. | Always check if the receptacle is closed first.                       |

## 3. Fallback Search Algorithm
If the target is not found at the primary location, execute this expanded search pattern.

