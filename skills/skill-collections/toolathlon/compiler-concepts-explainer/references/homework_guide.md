# Homework Guide: Symbol Tables
*Based on "HW.pdf" Assignment*

## Question 1: Scope Restoration Methods
**Topic:** Efficiency of different symbol table implementations when exiting a scope.
*   **A (Deep Copy):** Inefficient. Copies entire data structure.
*   **B (Functional BST):** Efficient. Creates only O(d) new nodes for insertion at depth d. Original is preserved.
*   **C (Imperative Hash + Undo Stack):** Efficient. Restoration involves pointer manipulation on the undo stack.
*   **Correct Answer:** **D (Both B and C)**. Both methods achieve restoration with minimal overhead compared to a full copy.

## Question 2: Variable Binding & Shadowing
**Topic:** Applying scope rules to determine a variable's active binding.
**Given Code:** Function `f` with parameters `a:int, b:int, c:int` and an inner `let` block that declares `var a := "hello"`.
**Analysis:**
1.  Outer scope (`σ1`): `a ↦ int` (parameter).
2.  Inner `let` scope (`σ3`): A new binding `a ↦ string` is added, **shadowing** the parameter.
3.  `print(a)` inside the `let` block looks up `a` in the current environment (`σ3`), finding the `string` binding.
**Correct Answer:** **string**.

## Question 3: Hash Table Insertion Strategy
**Topic:** How imperative hash tables handle duplicate identifiers (shadowing).
*   The imperative style uses a stack-like discipline for scoping.
*   To shadow a binding `a ↦ τ1` with `a ↦ τ2`, the new binding is inserted **at the head** of the hash chain.
*   Lookup finds the new binding first. `pop()` removes it, restoring the old one.
*   **Correct Answer:** **C (Insert the new binding at the head of the collision chain)**.

## Question 4: String Optimization
**Topic:** The technique used to avoid expensive string operations in symbol tables.
*   The problem is repeated character-by-character hashing and comparison.
*   The solution is to convert each distinct string to a canonical **symbol** object (`S_symbol`).
*   Operations then use the symbol's address (fast integer/pointer ops).
*   **Correct Answer:** **Symbol** (or `S_symbol`).

## Question 5: Forward References
**Topic:** Managing multiple symbol tables in languages with mutual recursion (e.g., Java).
*   Java allows forward references (class D can use class N defined later).
*   Each class (`E`, `N`, `D`) has its own local symbol table for its members.
*   These tables are combined into a single, comprehensive environment (`σ7`) for the whole package/compilation unit.
*   All classes are compiled with access to this combined environment.
*   **Correct Answer:** **B (Each class has its own symbol table, which are combined into a comprehensive environment)**.
