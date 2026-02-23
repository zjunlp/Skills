---
name: scienceworld-material-classifier
description: This skill makes a determination about a material's property (e.g., conductivity) based on environmental cues or domain knowledge when direct testing fails. Trigger it when experimental actions are invalid or unavailable, requiring a logical inference. It uses observed object properties and common-sense reasoning to classify the material and decide its final disposition.
---
# Material Classification Skill

## When to Use
Activate this skill when:
1. Direct experimental testing of a material property (conductivity, magnetism, etc.) fails due to invalid actions or unavailable equipment.
2. You need to make a logical inference based on observed properties or domain knowledge.
3. A final disposition decision (e.g., placing in specific container) is required.

## Core Logic
1. **Identify Material**: Focus on the target object and note its composition (e.g., "glass jar").
2. **Attempt Direct Testing**: First try standard experimental actions if equipment exists (e.g., connecting to circuit).
3. **Fallback to Inference**: When direct testing fails:
   - Consult material properties in `references/material_properties.md`
   - Use common-sense reasoning (e.g., glass is typically nonconductive)
   - Consider observed contents (e.g., sodium chloride is conductive but container material dominates)
4. **Execute Disposition**: Place the object in the appropriate container based on classification.

## Critical Notes
- Glass containers are generally electrical insulators regardless of contents
- Metal objects are typically conductive unless specified otherwise
- When in doubt, use the most common material property from domain knowledge
- Always verify the target object is properly identified before classification
