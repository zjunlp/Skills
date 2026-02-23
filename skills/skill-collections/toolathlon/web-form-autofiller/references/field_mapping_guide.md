# Field Mapping Guide for Common Form Types

This guide helps map entity attributes from a knowledge graph to typical web form fields.

## Generic Attribute to Form Field Mapping

| Entity Attribute (From Observations) | Typical Form Field Type | Mapping Logic & Notes |
| :--- | :--- | :--- |
| `Name: [Full Name]` | Text Input | Direct mapping. |
| `Email: [email]` | Text Input | Direct mapping. |
| `Address: [address]` | Text Input | Direct mapping. |
| `Phone: [number]` | Text Input | Direct mapping. |
| `Student ID: [id]` | Text Input | Direct mapping. |
| `Birthday: [YYYY-MM-DD]` | Date Input | Use standard YYYY-MM-DD format. |
| `Major: [field]` | Text Input / Dropdown | May need to match specific options. |
| `Hobbies: [list]` | Checkbox / Radio Group | Map to specific activity options. For single-select (radio), choose the most relevant. |
| `Health condition: [condition]` | Text Area / Radio Group | e.g., "Gout, cannot eat seafood" -> maps to "No Seafood" dietary restriction. |
| `Dietary restrictions: [list]` | Radio Group / Checkbox | Map to provided options (None, Vegan, Kosher, No Seafood, etc.). |
| `Mental health: [description]` | Radio Group (e.g., Anxiety) | If description is positive ("Healthy", "Good"), default to low anxiety (1). |
| `Degree: [level]` | Radio Group | Map "Bachelor's" -> "bachelor", "Master's" -> "master", etc. |
| `Currently pursuing: [degree]` | (Usually not mapped) | **Ignore for "Highest degree earned" field.** |
| `GPA: [score]` | Text Input / Range | May need formatting. |
| `Swimming ability: [ability]` | Radio Group (Activities) | "Cannot swim" means do NOT select the "swimming" activity. |

## Handling Common Form Patterns

### 1. Session/Time Selection (Checkboxes)
-   **User Request:** "participate for the whole day"
-   **Action:** Select **all** relevant checkboxes (e.g., "Morning" AND "Afternoon").

### 2. Single-Choice Questions (Radio Groups)
-   **Rule:** Select exactly one option.
-   **Data Conflict:** If entity data suggests multiple fits (e.g., hobbies include "Programming" and "Basketball"), you must **choose one**. Use context:
    -   Prefer an activity explicitly listed in the form's options.
    -   If "cannot [activity]" is stated, exclude that option.
    -   Default to the first logically matching option.

### 3. Default Values for Missing Data
-   **Principle:** Default to the most "negative", "neutral", or "low" option.
-   **Anxiety Scale (1-5):** Default to **1** (low anxiety).
-   **Yes/No or Positive/Negative:** Default to "No" or the negative option.
-   **Optional Fields:** Leave blank if no data and no default is specified.

### 4. Form Navigation
-   Large forms may be split across multiple "spans" in the snapshot view.
-   Use `browser_snapshot_navigate_to_next_span` and `browser_snapshot_navigate_to_first_span` to view all fields before deciding on a mapping strategy.
