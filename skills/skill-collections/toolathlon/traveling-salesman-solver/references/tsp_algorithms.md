# Traveling Salesman Problem (TSP) - Algorithms & Considerations

## Problem Definition
Given a list of locations and the distances between each pair, find the shortest possible route that visits each location exactly once and returns to the origin.

## Common Algorithms

### 1. Brute Force (Exhaustive Search)
*   **Method:** Evaluate all possible permutations of locations.
*   **Complexity:** O(n!) - Factorial time.
*   **Use Case:** Optimal solution for n ≤ 10 locations.
*   **Implementation:** Use `itertools.permutations` in Python.

### 2. Nearest Neighbor (Greedy Heuristic)
*   **Method:** Start at an arbitrary location, repeatedly visit the nearest unvisited location.
*   **Complexity:** O(n²) - Much faster.
*   **Use Case:** Quick, approximate solution for large n. Not guaranteed optimal.
*   **Accuracy:** Typically 10-25% longer than optimal.

### 3. Held-Karp (Dynamic Programming)
*   **Method:** DP algorithm that solves exact TSP in O(n²2ⁿ) time.
*   **Use Case:** Optimal solution for n ≤ 20-25 (due to memory constraints).

## Practical Considerations for Route Planning

### Asymmetric vs. Symmetric Distances
*   **Walking/Driving:** Usually symmetric on road networks, but one-way streets can create asymmetry.
*   **Public Transit:** Often highly asymmetric due to schedules.
*   **Default Assumption:** Symmetric unless specified.

### Start and End Points
*   **Classic TSP:** Returns to start (closed tour).
*   **Open TSP:** Ends at a different location.
*   **Fixed Start:** Common in tourist itineraries (start at hotel).

### Real-World Constraints
*   **Time Windows:** Locations may have opening hours.
*   **Capacity:** Vehicle routing with load limits.
*   **Multiple Vehicles:** VRP (Vehicle Routing Problem).

## Distance Matrix Properties
A valid distance matrix should be:
1.  **Square:** n × n for n locations.
2.  **Non-negative:** All distances ≥ 0.
3.  **Zero Diagonal:** Distance from a location to itself is 0.
4.  **Symmetric (usually):** distance[i][j] = distance[j][i].

## Optimization Tips
1.  **Cluster First:** Group nearby locations before solving.
2.  **Use Heuristics for n > 10:** Nearest Neighbor, Christofides, or Lin-Kernighan.
3.  **Consider Time vs. Distance:** Sometimes shorter distance ≠ shorter time (traffic, terrain).
4.  **Validate with Mapping APIs:** Network distance ≠ straight-line distance.
