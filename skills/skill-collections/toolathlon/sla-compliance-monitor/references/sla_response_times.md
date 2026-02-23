# SLA Response Time Standards
Extracted from MCP Inc. Service Standards & Operations Handbook (June 2024)

## User Levels and Response Times

| User Level | First Reply Time | Second Reply Time (if first reply is overdue) |
|------------|------------------|-----------------------------------------------|
| Basic      | 72 hours         | 72 hours                                      |
| Pro        | 36 hours         | 36 hours                                      |
| Max        | 24 hours         | 18 hours                                      |

## Definitions
- **First Reply Time**: Maximum time from ticket creation to the first customer contact
- **Second Reply Time**: If the first reply exceeds the deadline, you must send an apology email stating that the next reply will be sent within this time frame

## Notes
- These times apply only to tickets where `FIRST_RESPONSE_AT` IS NULL
- Response time calculation: `TIMESTAMPDIFF(HOUR, CREATED_AT, CURRENT_TIMESTAMP())`
- Tickets should be prioritized in order: Max → Pro → Basic
