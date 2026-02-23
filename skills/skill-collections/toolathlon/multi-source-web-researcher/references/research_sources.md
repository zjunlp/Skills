# Research Sources & Verification Guide

## Primary Source Hierarchy
For academic research tasks, verify information in this order:
1.  **Original Paper PDF**: Ground truth for author list, affiliations, and correspondence.
2.  **Official Repository Page**: Conference sites (OpenReview, PMLR), arXiv abstract pages.
3.  **Author's Institutional Profile**: University department pages, lab websites.
4.  **Professional Network Profiles**: Google Scholar, LinkedIn, ResearchGate.
5.  **General Web Search Results**: News articles, blog posts, secondary summaries.

## Common Source Patterns

### 1. Finding Paper PDFs
- **arXiv**: `https://arxiv.org/pdf/{paper_id}` (e.g., `2502.12147`)
- **OpenReview (ICML/NeurIPS/ICLR)**: `https://openreview.net/pdf?id={paper_id}`
- **PMLR (Proceedings of Machine Learning Research)**: `https://proceedings.mlr.press/v{volume}/{paper_id}.pdf`

### 2. Extracting Author Information
From a PDF's first page, look for:
- The title and author list directly under it.
- The affiliation block, often marked with numbered superscripts (`¹`, `²`) or asterisks (`*`).
- A "Correspondence to:" line for the corresponding author's email and institution.

### 3. Finding Google Scholar Profiles
Search query patterns:
- `"<Full Author Name>" Google Scholar`
- `"<Author Name>" site:scholar.google.com`
- Combine with institution: `"<Author Name>" "<Institution>" Google Scholar`

Profile URL pattern: `https://scholar.google.com/citations?user={USER_ID}&hl=en`

### 4. Verifying Affiliations
- Extract the *exact* affiliation string as it appears in the paper.
- Include all institutions if multiple are listed.
- Include department-level information when provided (e.g., "Department of Computer Science").
- Do not abbreviate unless the paper does.

## Concurrent Search Strategy
When researching multiple items (e.g., 6 papers):
1.  Launch all initial web searches for "first author" in parallel.
2.  From results, identify the most promising PDF source URLs for each.
3.  Launch all PDF reads in parallel to extract affiliations.
4.  Launch all Google Scholar profile searches in parallel.
5.  Consolidate results and write to target file.
