---
name: notion-project-notification-manager
description: When the user needs to update Notion pages with project status, completion notifications, or resource links, this skill searches for specific Notion pages, adds content updates, and posts comments or notifications. It can locate pages by title, add formatted text, and include external resource links. Triggers include requests to 'write on Notion page', 'add comment to page', 'update Notion', or notifications about task completion.
---
# Skill: Notion Project Notification Manager

## Primary Objective
Search for a specified Notion page and update it with a status notification and/or a link to an external resource (e.g., a Google Sheet).

## Core Workflow
1.  **Parse the Request:** Identify the target Notion page name/title and the specific update content (notification text and resource URL) from the user's request.
2.  **Locate the Page:** Use the Notion Search API to find the page by its title. Filter the search to object type `"page"` for accuracy.
3.  **Prepare the Update:** Format the update content. This typically involves:
    *   A clear status notification (e.g., "Monthly market data is ready. The reporting team can view it directly").
    *   The external resource link on a new line (e.g., `Google Sheet : {url}`).
4.  **Execute the Update:** Append the formatted content to the target Notion page. Use the appropriate Notion API endpoint for adding blocks or text to a page.

## Key Instructions & Logic
*   **Search Specificity:** Always use a filter `{"property":"object","value":"page"}` when searching for a page by name to avoid confusion with databases.
*   **Content Formatting:** Structure the update clearly. Place the status notification first, followed by the resource link on a separate line.
*   **Error Handling:** If the initial search returns multiple results or no results, refine the search query or ask the user for clarification.
*   **Idempotency:** The skill should be able to run multiple times without causing duplicate content if the same notification is specified. Consider checking existing page content if necessary.

## Common Triggers
*   "Write on the [Page Name] Notion page..."
*   "Add a comment to the [Page Name] page stating..."
*   "Update the Notion page for [Project] with a link to..."
*   "Notify the team on Notion that [Task] is complete."

## Notes
*   This skill focuses on the Notion update operation. The creation or population of external resources (like Google Sheets) is handled by other skills or prior steps in the agent's workflow.
*   The example trajectory shows the final step of notifying a "Quant Research" page after data has been compiled elsewhere.
