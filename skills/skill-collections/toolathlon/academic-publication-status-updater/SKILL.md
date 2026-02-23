---
name: academic-publication-status-updater
description: When the user needs to update their academic homepage with recent paper acceptance information and code repository links. This skill scans email notifications for paper acceptance confirmations, identifies papers marked as 'preprint' or 'under review' on the user's GitHub-hosted homepage, updates their status to 'accepted', and adds corresponding GitHub repository links for code open-sourcing. It handles conference notifications (COML, COAI, COLM, etc.), workshop papers, and oral presentation selections. Triggers include requests to 'update homepage papers', 'add acceptance information', 'link code repositories', or 'sync paper status from emails'.
---
# Instructions

## Goal
Update the user's academic homepage (hosted on GitHub) by:
1.  Identifying papers currently marked as "preprint" or "under review".
2.  Checking the user's emails for acceptance notifications for those papers.
3.  Updating the paper's status on the homepage from "under review" to "accepted" (including specific conference/venue details).
4.  For accepted papers, checking if a corresponding code repository exists on the user's GitHub and adding a link to it on the homepage.

## Core Workflow

### 1. Initial Discovery & Setup
*   Use `github-get_me` to confirm the user's GitHub identity and profile.
*   Use `github-search_repositories` with the user's username to find their homepage repository (typically named something like `My-Homepage`, `personal-website`, `academic-kickstart`, or containing `_publications/` or `_posts/` directories). Also note other repositories that might contain paper code.
*   Use `emails-get_emails` or `emails-search_emails` to fetch recent emails. Look for keywords like "accepted", "camera-ready", "congratulations", and conference names (COML, COAI, ICML, NeurIPS, etc.).

### 2. Analyze Homepage Publications
*   Navigate to the homepage repository's `_publications/`, `_posts/`, or similar directory using `github-get_file_contents`.
*   Read the Markdown or YAML files for each publication.
*   **Identify Target Papers**: Parse the `venue` field. Flag any paper where the venue contains phrases like "Under review at", "preprint", "arXiv", or "submitted to".

### 3. Cross-Reference with Emails
*   For each target paper, extract its title and suspected conference name.
*   Search through the fetched emails for matches. **Key email indicators**:
    *   Subject lines containing "accepted", "camera-ready", or the conference name.
    *   Email body containing the paper's title or clear congratulations.
    *   Specific details like "oral presentation" or "workshop".
*   Use `emails-read_email` on promising emails to get full details and confirm acceptance.

### 4. Map Code Repositories
*   From the initial repository search, create a mapping. Repository names often contain keywords from paper titles (e.g., "llm-adaptive-learning", "optimizing-llms-contextual-reasoning").
*   For each **accepted** paper, try to find a matching repository by comparing sanitized paper titles/keywords with repository names.

### 5. Execute Updates
*   For each accepted paper, prepare an updated publication file.
    *   Change the `venue` field from "Under review at X" to "X" or "X (Oral)".
    *   **If a matching code repository was found**, add or update a `codeurl:` field in the frontmatter with the GitHub URL (e.g., `https://github.com/<username>/<repo-name>`).
*   Use `github-get_file_contents` to get the current file's SHA.
*   Use `github-create_or_update_file` to commit the changes with a descriptive message (e.g., "Update paper status: [Paper Title] accepted at [Conference], add code repository link").

## Key Decision Points & Heuristics
*   **Conference Name Extraction**: When updating the `venue` field, remove "Under review at " or "Submitted to ". Use the full conference name from the email (e.g., "COML 2025 - Conference on Machine Learning").
*   **Oral Presentations**: If an email specifies "oral presentation", append " (Oral)" to the venue.
*   **Workshop Papers**: Treat workshop acceptances (e.g., COMLW) as standard acceptances.
*   **Code Repository Matching**: Be flexible. Match on key terms (ignore "the", "for", "a", hyphens, spaces). A repository like `optimizing-llms-contextual-reasoning` clearly matches a paper titled "Optimizing Large Language Models for Contextual Reasoning...".
*   **Email Search Strategy**: If an initial targeted search (`"paper accepted"`) fails, fall back to fetching a broader batch of emails and scanning them manually.

## Error Handling & Validation
*   **SHA Mismatch**: If `github-create_or_update_file` fails with a 409 error, refetch the file to get the latest SHA and retry.
*   **No Homepage Found**: If no obvious homepage repo is found, list the user's repositories and ask for clarification.
*   **Ambiguous Email**: If an email's acceptance is not explicit, do not update the paper. It's better to be conservative.
*   **No Matching Code Repo**: It's acceptable. Only add the `codeurl` if a clear match is found.

## Final Output
Provide the user with a summary of changes made:
*   List each paper updated, its new status, and conference.
*   List any code repository links added.
*   Note any papers that remain under review because no acceptance email was found.
