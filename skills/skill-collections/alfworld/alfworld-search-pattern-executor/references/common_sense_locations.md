# Common Sense Location Priorities

This reference provides typical search priorities for household objects, derived from general world knowledge. Use this to generate the initial `candidate_receptacles` list for the `alfworld-search-pattern-executor` skill.

## High-Priority Locations (Search First)
*   **Surfaces near seating/entertainment:** Coffee tables, side tables, sofas, armchairs, TV stands.
*   **General storage surfaces:** Dressers, countertops, desks, shelves.
*   **Designated storage:** Cabinets, drawers (especially in living rooms or entertainment centers).
*   **Containers:** Boxes, baskets, bins in relevant rooms.

## Object-Specific Heuristics

### Remote Controls
1.  Coffee Table
2.  Sofa / Armchair (cushions)
3.  Side Table
4.  TV Stand / Entertainment Center
5.  Drawers in living room furniture
6.  Countertops (kitchen, if misplaced)

### Keys / Keychains
1.  Key hooks / entryway table
2.  Drawers near entrance
3.  Countertops
4.  Bags / purses
5.  Dresser tops

### Books / Magazines
1.  Bookshelves
2.  Coffee Tables
3.  Nightstands
4.  Desks
5.  Baskets or magazine racks

### Kitchen Utensils
1.  Utensil drawers
2.  Countertop holders
3.  Dishwasher
4.  Sink
5.  Cabinets

## Search Pattern Principles
1.  **Visible First:** Always check surfaces and open areas before enclosed storage.
2.  **Proximity:** Search areas logically associated with the object's use (e.g., remotes near TVs).
3.  **Frequency:** Check commonly used locations first, then less frequented ones.
4.  **Room-by-Room:** If the environment is large, constrain the search to the most relevant room first.

## Example Candidate Lists
*   **For a remote in a living room:** `['coffeetable 1', 'sofa 1', 'sidetable 1', 'dresser 1', 'drawer 1', 'drawer 2', 'cabinet 1']`
*   **For a book in a bedroom:** `['dresser 1', 'sidetable 1', 'desk 1', 'bookshelf 1', 'drawer 1']`
