# WebShop Action Protocol Reference

## Overview
This document details the strict action-response protocol used in the WebShop environment that this skill operates within.

## Action Format
All actions must be output in one of two formats:

1. **Search Action:** `search[<keywords>]`
   *   `<keywords>`: A string of search terms. These should be carefully chosen to match the user's request.
   *   **Usage:** Primarily used before this skill is triggered. This skill focuses on post-search navigation.

2. **Click Action:** `click[<value>]`
   *   `<value>`: **Must exactly match** one of the strings present in the "clickables" list from the most recent observation.
   *   **Case Sensitivity:** The match must be exact, including capitalization and spacing.
   *   **Validation:** If the value is not in the list, the action is invalid and nothing happens.

## Response Format
The agent's response must follow this exact two-line structure:
