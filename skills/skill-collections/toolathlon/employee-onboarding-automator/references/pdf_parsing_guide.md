# PDF Parsing Guide

## Document: landing_tips.pdf
This PDF contains the company's unified onboarding checklist.

## Key Sections to Extract

### 1. Public Tasks (All Departments)
Look for the section titled "Public Tasks (All Departments, complete by Day 7)".
Extract these tasks in order:
- Onboarding Training
- Security Training
- Confidentiality Training
- Company Culture
- Company Strategy

### 2. Group-Specific Tasks
Extract tasks for each group with their relative deadlines:

#### Backend Group Tasks (complete by Day 30)
- Backend Development Process
- Backend Development Standards
- Backend Development Environment

#### Frontend Group Tasks (complete by Day 45)
- Frontend Development Process
- Frontend Development Standards
- Frontend Development Environment

#### Testing Group Tasks (complete by Day 60)
- Testing Development Process
- Testing Development Standards
- Testing Development Environment

#### Data Group Tasks (complete by Day 75)
- Data Development Process
- Data Development Standards
- Data Development Environment

## Parsing Notes
- The PDF may contain Unicode characters (e.g., `\u0000` prefixes). Clean the text by removing null characters.
- "D+N" means "N calendar days after Day 0" (the start date).
- Focus on the task names and their relative deadlines. Ignore supplementary notes and descriptions.
