# CSRankings Interface Guide

## URL Parameters & Filters
CSRankings uses a hash-based URL system to represent state. Key parameters include:

*   `index?ai&vision&mlmining&nlp`: Selects AI research areas (Artificial Intelligence, Computer Vision, Machine Learning, Natural Language Processing).
*   `&us`: Filters to institutions in the United States.
*   `fromyear/2024/toyear/2024`: Sets the publication year range (both must be set for a single year).

## Page Structure & Data Extraction
1.  **Loading:** The page shows "Loading publication data." initially. Wait for "Rank took..." console messages.
2.  **Filter Area:** A left sidebar contains expandable sections for research areas. Checkboxes indicate selection.
3.  **Ranking Table:** The main content is a table with columns:
    *   `#`: Numerical rank.
    *   `Institution`: University name and flag icon.
    *   `Count`: Normalized publication score.
    *   `Faculty`: Number of contributing faculty.
4.  **Ties:** Identical ranks appear as duplicate numbers in consecutive rows.

## Important Checkboxes for AI (Excluding Information Retrieval)
For a comprehensive AI ranking that excludes Information Retrieval, ensure the following checkboxes are **checked**:
-   [x] Artificial intelligence
-   [x] Computer vision
-   [x] Machine learning
-   [x] Natural language processing

And this checkbox is **unchecked**:
-   [ ] The Web & information retrieval

## Year Selection
Use the two dropdowns labeled "by publications from" and "to". For a ranking of publications in 2024 only, set both to "2024".
