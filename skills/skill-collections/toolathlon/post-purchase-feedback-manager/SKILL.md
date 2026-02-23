---
name: post-purchase-feedback-manager
description: When the user needs to collect customer feedback after completed purchases, this skill manages the entire workflow retrieves completed orders from WooCommerce, creates customized Google Forms based on requirements specifications, extracts customer emails, sends personalized survey invitations, and stores form links. It triggers on requests involving customer feedback collection, post-purchase surveys, satisfaction measurement, or when working with WooCommerce order data and Google Forms integration.
---
# Instructions

## Workflow Overview
Execute this skill when the user requests to collect feedback from customers with completed purchases. The skill orchestrates a multi-step process involving WooCommerce, Google Forms, and email systems.

## Step-by-Step Execution

### 1. Retrieve Completed Orders
- Use the `woocommerce-woo_orders_list` tool to fetch all orders with status "completed".
- Set `perPage` to 100 (or adjust based on expected volume).
- Extract the JSON response containing order details.

### 2. Locate Requirements File
- Check for a feedback form requirements file in the workspace.
- Common filenames: `form_requirement.md`, `form_requirements.md`, `feedback_spec.md`.
- If not found, list the workspace directory to identify the correct file.
- Read the requirements file to understand the survey structure.

### 3. Create Google Form
- Use `google_forms-create_form` with the title specified in requirements.
- Note: Only the title can be set during creation; description and items are added separately.
- Capture the returned `formId` and `responderUri` for later use.

### 4. Add Survey Questions
Based on the requirements file, add questions using appropriate Google Forms tools:
- **Multiple choice questions**: Use `google_forms-add_multiple_choice_question`
- **Text questions**: Use `google_forms-add_text_question`
- **Star ratings**: Implement as multiple choice with descriptive options (e.g., "1 star - Very dissatisfied" to "5 stars - Very satisfied")
- Set `required` field according to specifications.

### 5. Extract Customer Emails
- Parse the WooCommerce orders JSON to extract unique customer email addresses.
- Each order contains billing information with `email` field.
- Note: Some customers may have multiple completed orders; deduplicate emails.
- Extract customer names from `billing.first_name` and `billing.last_name` for personalization.

### 6. Send Personalized Invitations
- For each unique customer email:
  - Use `emails-send_email` tool
  - Subject: "We Value Your Feedback - Customer Shopping Experience Survey"
  - Body: Personalized greeting using customer name, thank them for their purchase, include the survey link (`responderUri`), and polite closing.
  - Send individually to maintain personalization.

### 7. Store Form Reference
- Save the Google Drive link to a workspace file (e.g., `drive_url.txt`).
- Format: `https://drive.google.com/open?id={formId}`
- Use `filesystem-write_file` to create/update the file.

## Error Handling
- If requirements file not found: List directory contents and try common variations.
- If form creation fails: Check error message and adjust parameters.
- If email sending fails: Continue with remaining emails and note the failure.

## Completion
- After all steps complete, provide a summary including:
  - Number of completed orders processed
  - Number of unique customers contacted
  - Google Form ID and URL
  - Location of stored drive link
- Claim completion using `local-claim_done`.
