# Notion Database Schema Reference

## Job Tracker Database
**Expected Properties:**
- `Company` (title): Name of the company.
- `Position` (url): Job position title.
- `Status` (select): Application status with options: Checking, Applied, Interviewing, Rejected, Pass, Accepted.
- `Salary Range` (rich_text): Salary information in text format (e.g., "$4200 - $4800/mo").
- `Location` (rich_text): Company location (e.g., "Los Angeles, US").
- `Email` (rich_text): Contact email address.
- `Flexibility` (select): Work arrangement with options: Remote, On-site.
- `In-touch Person` (rich_text): Contact person name.

## API Usage Notes
1. **Searching for Database**: Use `notion-API-post-search` with query "Job Tracker" or page name.
2. **Querying Entries**: Use `notion-API-post-database-query` with filter on `Status` property.
3. **Updating Status**: Use `notion-API-patch-page` with properties: `{"Status": {"select": {"name": "Applied"}}}`.

## Field Access Patterns
- Title properties: `properties.Company.title[0].text.content`
- Rich text properties: `properties.Location.rich_text[0].text.content`
- Select properties: `properties.Status.select.name`
- URL properties: `properties.Position.url`
