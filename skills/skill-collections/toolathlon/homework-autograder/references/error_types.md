# Common Python Error Types in Student Submissions

When grading programming assignments, you will encounter these common errors. Use this reference to identify and provide accurate feedback.

## 1. Syntax Errors
The code cannot be parsed by Python.
- **Missing colon:** `for i in range(10)  # missing colon`
- **Missing parenthesis/bracket:** `print("hello"` or `list = [1, 2, 3`
- **Incorrect indentation**
- **Misspelled keywords:** `def` vs `deff`

**Feedback example:** "SyntaxError: missing colon at the end of the for statement on line 6."

## 2. Runtime Errors
The code parses but fails during execution.
- **NameError:** Using a variable that hasn't been defined.
- **TypeError:** Operating on incompatible types (e.g., `"string" + 5`).
- **IndexError:** Accessing a list index that doesn't exist (e.g., `list[10]` for a 3-item list).
- **KeyError:** Accessing a dictionary key that doesn't exist.
- **ModuleNotFoundError:** Importing a module that isn't installed or doesn't exist.
- **ZeroDivisionError:** Dividing by zero.

**Feedback example:** "IndexError: list index out of range. Your loop goes to len(nums)+1 instead of len(nums)."

## 3. Logic Errors
The code runs without errors but produces incorrect output.
- **Wrong algorithm:** Using an incorrect approach to solve the problem.
- **Off-by-one errors:** Loops running one time too many or too few.
- **Incorrect output format:** Returning results in wrong order (e.g., `[1, 0]` instead of `[0, 1]`).
- **Missing edge cases:** Not handling specific inputs mentioned in the assignment.

**Feedback example:** "Logic error: Your solution returns indices in descending order [1, 0], but the requirement specifies ascending order [0, 1]."

## 4. Style/Requirement Violations
The code works but doesn't follow assignment specifications.
- **Wrong function name:** Using `two_sum` instead of `twoSum`.
- **Missing function signature:** Not using the required parameters.
- **Hard-coded values:** Returning a fixed answer instead of computing it.
- **External libraries:** Using modules not allowed by the assignment.

**Feedback example:** "Used external module 'non_existent_module' which is not allowed per assignment requirements."

## Grading Decision Tree
1. Does the code have a SyntaxError? → Score: 0
2. Does the code have a RuntimeError when executed? → Score: 0
3. Does the code produce the EXACT expected output for all test cases? → If NO, Score: 0
4. Only if all above pass → Score: 10 (full points)
