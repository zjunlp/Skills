---
name: academic-profile-publication-updater
description: When the user needs to update their academic profile website with new publications, particularly when they mention specific research papers (like arXiv articles) that need to be added to their publications list. This skill handles the complete workflow analyzing the existing website structure, searching for paper details in arXiv/local storage, understanding the publication format used by the site (Jekyll-based academic pages), and properly formatting new entries to match the existing style with correct author formatting, venue information, links, and bullet-pointed contributions. It's triggered by requests involving 'update profile', 'add publications', 'upload articles', or when users mention specific paper titles they want to include in their academic portfolio.
---
# Instructions

## 1. Understand the Request
- The user will mention they have uploaded or want to add specific publications to their academic profile.
- Identify the paper titles or arXiv IDs mentioned (e.g., "B-STaR", "SimpleRL-Zoo").
- Clarify if they want to update a specific file (like `about.md`) or the publications collection.

## 2. Analyze the Existing Website Structure
- First, navigate to the user's profile website URL (usually derived from the repository name or config).
- Use the browser tool to snapshot the live site and examine the **current publications section**.
- Pay close attention to:
  - The exact Markdown formatting used (headings, bold titles, line breaks).
  - How author lists are formatted (especially co-first author notation `*` and underlining `<ins>`).
  - The structure of venue/year information.
  - How links are formatted (`[[arxiv]]`, `[[github]]`, etc.).
  - How bullet-pointed contributions are written.

## 3. Locate the Local Repository
- Use `filesystem-list_directory` to explore the workspace.
- Look for common academic website structures (Jekyll, Hugo, etc.).
- Identify the main content directory (often `_pages`, `content`, or root).
- Find the file that needs updating (commonly `about.md`, `index.md`, or `_pages/publications.md`).

## 4. Search for Paper Details
- Use `arxiv_local-search_papers` with the paper titles or relevant keywords.
- If not found locally, use `arxiv_local-download_paper` with known arXiv IDs.
- Extract key details: **title**, **authors**, **abstract**, **published date**, **arXiv URL**, **categories**.
- If the paper is not on arXiv, use `local-web_search` to find official publication venues (e.g., "ICLR 2025", "COLM 2025").

## 5. Examine the Local File's Current State
- Read the target Markdown file.
- Compare its content with the **live website snapshot**. The live site is the source of truth for formatting.
- Note any discrepancies (the local file may be outdated).

## 6. Format the New Publication Entry
**CRITICAL:** Match the **exact formatting** observed on the live website.
General pattern observed:
