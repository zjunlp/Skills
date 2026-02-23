# Validation Analysis Patterns

## Common Analysis Scenarios

### 1. Identifying Best Experiment
**Goal:** Find which run has the highest validation performance
**Approach:**
1. Extract validation scores from `summaryMetrics` of all runs
2. Compare final validation scores
3. Consider runs that may have crashed or are still running

**Edge Cases:**
- Some runs may have null validation scores
- Validation metric names may vary
- Some projects may not log validation scores to summaries

### 2. Finding Optimal Step Within a Run
**Goal:** Identify the training step with peak validation performance
**Approach:**
1. Fetch full history for the best run
2. Extract validation scores at each step
3. Find maximum validation score and corresponding step

**Patterns to Consider:**
- Early stopping: Best score may not be at final step
- Overfitting: Validation score may decline after peak
- Evaluation frequency: Validation may be logged at intervals, not every step

### 3. Metric Name Discovery
**Strategies for identifying validation metrics:**
1. Check common names: `val/`, `validation/`, `eval/`, `test_`
2. Examine `historyKeys` from run query for metric patterns
3. Look for metrics with "score" or "accuracy" in name
4. Check if metrics have fewer data points (evaluation happens less frequently)

## Data Processing Patterns

### Handling Large History Data
1. **Sampling:** Use `sampledHistory` with specs for large runs
2. **Pagination:** Handle paginated results for very long runs
3. **File Storage:** Large responses may be saved to files; use search tools

### Parsing JSON Strings
History data comes as JSON strings within JSON:
