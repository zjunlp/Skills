# Living Entity Indicators Reference

This document lists terms and patterns that signal a potential living thing in the ScienceWorld environment. Use this list to update the `LIVING_INDICATORS` in the analysis script.

## High-Confidence Indicators
These terms almost always refer to living entities or their direct products:
*   **Eggs:** `egg`, `spawn`, `embryo`
*   **Plants:** `plant`, `flower`, `seed`, `sapling`, `tree`, `leaf`, `root`, `mushroom`, `fungus`
*   **Animals:** `animal`, `bird`, `fish`, `insect`, `mammal`, `reptile`, `amphibian`
*   **Specific Animals:** `dove`, `tortoise`, `butterfly`, `frog`, `mouse`, `rat`, `cat`, `dog`
*   **Life Stages:** `larva`, `pupa`, `tadpole`, `seedling`

## Medium-Confidence Indicators
These may be living or derived from living things:
*   `living`, `organism`, `biological`, `creature`, `being`
*   `nest`, `hive`, `web` (signs of animal activity)
*   `fruit`, `vegetable`, `nut` (plant reproductive parts)

## Low-Confidence/Ambiguous Indicators
These are often substances or non-living objects:
*   `air`, `water`, `soil` (substances)
*   `wood`, `bone`, `leather` (derived from living things but not alive)
*   `book`, `painting`, `chair` (inanimate objects)

## Environmental Context Clues
Rooms with higher probability of containing living things:
1.  **Outside:** Highest probability (eggs, plants, animals)
2.  **Greenhouse:** High probability (plants, seeds)
3.  **Bedroom/Living Room:** Low probability (pets, plants possible)
4.  **Kitchen/Workshop:** Very low probability (processed materials)

## Parsing Rules for Observation Text
1.  Objects listed as "a substance called X" are rarely living entities.
2.  Objects in parentheses "(containing...)" should be examined separately.
3.  Objects described with multiple words (e.g., "giant tortoise egg") should be treated as a single entity.
4.  Always prefer the most specific identifier (e.g., "dove egg" over just "egg").
