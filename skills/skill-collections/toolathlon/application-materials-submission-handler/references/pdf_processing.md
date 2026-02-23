# PDF Processing Guidelines

## Reading PDF Content
Use `pdf-tools-read_pdf_pages` to:
1. Extract professor names from recommendation letters (read first page)
2. Verify award certificate dates for sorting
3. Confirm document contents when filenames are ambiguous

## Merging PDFs
Use `pdf-tools-merge_pdfs` for:
- Combining multiple award certificates into `All_Awards_Certificates.pdf`
- Ensure proper sorting: chronological order (oldest to newest)

### Sorting Logic for Award Certificates:
1. Extract year from filename or content:
   - `Outstanding_Student_Award_2021.pdf` → 2021
   - `Research_Competition_First_Place_2022.pdf` → 2022
   - `Academic_Excellence_Award_2023.pdf` → 2023
2. Sort ascending: 2021, 2022, 2023
3. Merge in sorted order

## Quality Checks
- Verify all PDFs are readable (not corrupted)
- Confirm merged PDF contains correct number of pages
- Ensure text extraction works for key documents
