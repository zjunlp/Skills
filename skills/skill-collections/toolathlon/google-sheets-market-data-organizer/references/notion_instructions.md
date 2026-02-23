# Notion Page Update Instructions

## Target Page Identification
- Use the `notion-API-post-search` tool with a query for the exact page name (e.g., "Quant Research").
- Apply a filter to limit results to pages: `"filter": {"property":"object","value":"page"}`.
- From the results, identify the correct page by its `title` property and note its `id`.

## Required Updates
After successfully creating and populating the Google Sheet, make two updates to the identified Notion page:

1.  **Append a Line of Text:**
    - Use the appropriate tool (e.g., `notion-API-append-block`) to add a new text block to the page.
    - The text must be exactly: `Google Sheet : https://docs.google.com/spreadsheets/d/{spreadsheetId}`
    - Replace `{spreadsheetId}` with the actual ID from the `google_sheet-create_spreadsheet` response.

2.  **Add a Page Comment:**
    - Use the `notion-API-create-comment` tool.
    - The comment text must be exactly: `"Monthly market data is ready. The reporting team can view it directly"`
    - The `parent` parameter should be the page's ID.
