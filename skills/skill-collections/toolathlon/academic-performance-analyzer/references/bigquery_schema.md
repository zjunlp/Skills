# BigQuery Schema Reference

## Dataset: academic_warning

### Table Structure
All historical score tables follow the same schema:

**Table Name Pattern**: `scores_XXXX` (e.g., `scores_2501`, `scores_2502`)

**Columns**:
- `student_id` (STRING): Unique student identifier (e.g., "S001", "S002")
- `name` (STRING): Student's full name
- `score` (FLOAT): Test score (typically 0-100 scale)

### Example Table Names
Based on the execution trajectory, the dataset contains:
- `scores_2501`
- `scores_2502` 
- `scores_2503`
- `scores_2504`
- `scores_2505`
- `scores_2506`
- `scores_2507`

### Query Patterns

#### 1. Get All Historical Averages
