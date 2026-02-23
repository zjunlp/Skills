# Academic Profile Update Checklist

## Trigger Conditions
- [ ] User mentions "update profile", "add publication", "upload article"
- [ ] Specific paper titles or arXiv IDs are mentioned
- [ ] Request involves academic/portfolio website

## Step-by-Step Verification

### 1. Initial Analysis
- [ ] Identified target papers (B-STaR, SimpleRL-Zoo, etc.)
- [ ] Located user's website URL
- [ ] Determined which file needs updating (about.md, publications.md, etc.)

### 2. Research Phase
- [ ] Searched arXiv local storage for papers
- [ ] Downloaded missing papers if needed
- [ ] Extracted: title, authors, abstract, date, URL, venue
- [ ] Verified publication venue (ICLR, COLM, etc.)

### 3. Format Analysis
- [ ] Snapshot of live website publications section
- [ ] Examined exact formatting patterns:
  - [ ] Title formatting (bold, line breaks)
  - [ ] Author list (co-first *, user <ins>)
  - [ ] Venue/year format
  - [ ] Link formatting [[label]](url)
  - [ ] Bullet point style
- [ ] Compared local file vs live site

### 4. Content Creation
- [ ] Formatted title correctly
- [ ] Author list with proper * and <ins> tags
- [ ] Venue/year with period
- [ ] All relevant links included
- [ ] 3 bullet points summarizing contributions
- [ ] Used exact line break format (\\\\\\\\\\n)

### 5. File Update
- [ ] New entries placed at top (most recent first)
- [ ] Maintained existing formatting
- [ ] Preserved all original content
- [ ] No extra blank lines introduced
- [ ] For collections: created proper YAML frontmatter

### 6. Final Verification
- [ ] Re-read updated file
- [ ] Checked for Markdown syntax errors
- [ ] Confirmed only requested papers were added
- [ ] Provided summary to user

## Common Pitfalls to Avoid
- Using wrong line break format (must be \\\\\\\\\\n)
- Forgetting the period after venue/year
- Incorrect author formatting (missing * or <ins>)
- Adding extra publications not requested
- Breaking existing formatting in the file
- Using wrong venue abbreviation
