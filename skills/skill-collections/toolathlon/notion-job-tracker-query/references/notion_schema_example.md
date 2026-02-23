# Notion "Job Tracker" Database Schema Example

Based on the execution trajectory, the target database has the following properties. Use this as a reference, but always verify the actual schema from the API response.

## Property Map
| Property Name (UI) | Property ID (API) | Type (API) | Description & Notes |
| :--- | :--- | :--- | :--- |
| **Company** | `title` | `title` | The primary title property. Contains company name. |
| **Position** | `%60fZ%5B` | `url` | Stored as a URL string (e.g., "Software Engineer"). Treat as plain text. |
| **Status** | `a~~K` | `select` | Options: `Checking`, `Applied`, `Interviewing`, `Rejected`, `Pass`, `Accepted`. |
| **Salary Range** | `r_wX` | `rich_text` | Text like "$4200 - $4800/mo". Requires parsing for numeric values. |
| **Location** | `WvHp` | `rich_text` | City, Country (e.g., "Los Angeles, US"). Needs geocoding for distance calc. |
| **Email** | `%5E~%3B%40` | `rich_text` | Contact email address. |
| **In-touch Person** | `toSv` | `rich_text` | Name of the contact person. |
| **Flexibility** | `zSQ%3F` | `select` | Options: `Remote`, `On-site`. |

## API Response Structure for a Page
