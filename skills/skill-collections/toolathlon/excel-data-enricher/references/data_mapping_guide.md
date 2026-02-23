# Data Mapping Guide for Common Enrichment Tasks

This guide provides templates for mapping search queries and extraction logic based on common spreadsheet schemas.

## Schema 1: Academic Paper Research (from trajectory)
**Columns**: Title, First Author, Affiliation, Google Scholar Profile

### Research Strategy:
1.  **Query for First Author**: `"<Paper Title>" paper first author`
2.  **Query for Affiliation**: Access the PDF directly (via OpenReview, arXiv, etc.) and extract from the first page. Look for lines following author names, typically containing department and university names.
3.  **Query for Google Scholar**: `"<First Author Name>" Google Scholar profile` or `site:scholar.google.com "<First Author Name>"`

### Extraction Points (from PDF):
- **Author**: First name in the author list on page 1.
- **Affiliation**: Text on the same line or following line after the author's name, often marked with superscript numbers. Include all institutions if multiple are listed.
- **Example Format**: `Department of Computer Science, Purdue University` or `Gatsby Computational Neuroscience Unit, University College London`

## Schema 2: Company Contact Enrichment
**Columns**: Company Name, CEO, Headquarters, LinkedIn Page

### Research Strategy:
1.  **Query for CEO**: `"<Company Name>" CEO`
2.  **Query for HQ**: `"<Company Name>" headquarters location`
3.  **Query for LinkedIn**: `"<Company Name>" LinkedIn`

## Schema 3: Product Catalog Enrichment
**Columns**: Product ID, Product Name, Manufacturer, Manufacturer Website

### Research Strategy:
1.  **Query for Manufacturer**: `"<Product Name>" manufacturer`
2.  **Query for Website**: `"<Manufacturer Name>" official website`

## General Tips:
- **Prioritize Primary Sources**: When available, access the official document/page (PDF, company website, official profile).
- **Verify Consistency**: Ensure extracted names and affiliations match across sources.
- **Handle Multiple Results**: If a search returns multiple potential matches, use the most recent or most authoritative source (e.g., the paper's PDF for academic data).
