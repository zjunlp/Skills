# ScienceWorld Action Glossary for Measurement

## Core Measurement Actions
*   `use TOOL on TARGET`: The primary action for obtaining a measurement. Triggers the tool's function on the target and returns a reading in the observation.
*   `focus on OBJECT`: Signals intent on a task-critical object. Use this on the measurement tool and target after picking them up to confirm readiness.

## Navigation & Search Actions
*   `teleport to ROOM`: Instantly move to any named room. Use this to efficiently locate tools, targets, and move to the decision location.
*   `look around`: Describes the current room and lists all visible objects. The primary action for searching.

## Inventory Management Actions
*   `pick up OBJECT`: Moves a portable object from the environment into your inventory. Required before you can `use` or `focus on` it.
*   `move OBJECT to CONTAINER`: Places an object from your inventory into a container (e.g., a box). The key follow-up action after measurement.

## Conditional Logic in Observations
The observation from `use TOOL on TARGET` contains the measurement result. Parsing is environment-specific but typically follows patterns like:
*   `"the thermometer measures a temperature of 44 degrees celsius"`
*   `"the scale reads 120.5 grams"`

**Parsing Tip:** Use simple regex (e.g., `r'(\d+(?:\.\d+)?)'`) to extract the numeric value from the observation string.
