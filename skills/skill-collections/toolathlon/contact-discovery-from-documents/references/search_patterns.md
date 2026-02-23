# Search Patterns for Contact Discovery

## Recommended Search Queries (for `emails-search_emails`)
Start with these, adapting the `[keyword]`:
*   `[role]` (e.g., "cluster admin", "network", "billing")
*   `[system] admin` (e.g., "kubernetes admin", "database admin")
*   `[department] team` (e.g., "infrastructure team", "devops team")
*   `contact`
*   `responsible for`
*   `support`

## Common Document Filename Patterns (for `filesystem-search_files`)
*   `*contact*`
*   `*team*`
*   `*admin*`
*   `*guideline*`
*   `*policy*`
*   `*handbook*`
*   `*onboarding*`
*   `*config*` (especially in subdirectories like `k8s_configs/`)

## PDF Content Analysis Keywords
Scan extracted PDF text for these patterns:
*   **Section Headers:** `Contact`, `Team`, `Support`, `Responsibilities`, `Personnel`, `Administration`, `Owners`
*   **Table Indicators:** Look for aligned columns of `Name`, `Email`, `Role/Responsibility`.
*   **Inline Patterns:** `[Name] - [email] - [responsibility]`, `[Name] ([email])`, `[Responsibility]: [Name]`

## Directory Paths to Explore
*   `/workspace/dumps/workspace/` (common dump location)
*   `/shared/` or `/shared/configs/` (shared resources)
*   `/home/` or user directories
*   `/docs/` or `/documentation/`
