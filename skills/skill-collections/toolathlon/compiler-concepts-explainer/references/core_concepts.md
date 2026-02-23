# Core Compiler Concepts: Semantic Analysis & Symbol Tables
*Reference based on "Compile.pptx" Chapter 5*

## 1. Semantic Analysis Phase
**Purpose:** Bridge between syntax and machine code.
*   **Connects** variable definitions to their uses.
*   **Checks** type correctness of expressions.
*   **Translates** abstract syntax to a simpler IR.

## 2. Symbol Tables (Environments)
**Definition:** A data structure mapping identifiers (symbols) to their meanings (type, location).
*   **Binding:** Association `id ↦ meaning`, denoted by `σ = {a ↦ int, b ↦ string}`.
*   **Operations:**
    *   `enter(σ, id, binding)`: Add a new binding (`σ' = σ + {id ↦ binding}`).
    *   `lookup(σ, id)`: Find the current binding for `id`.
    *   Scope Exit: Discard bindings local to a finished scope.

## 3. The Two Implementation Styles

### Functional Style
*   **Core Idea:** Immutable environments. Creating a new environment (`σ'`) leaves the original (`σ`) intact and usable.
*   **Data Structure:** Balanced Binary Search Tree (BST) is efficient.
*   **Insertion (`σ' = σ + {a ↦ τ}`):** Creates new nodes *only* along the path from the root to the insertion point. Original tree is shared, not copied.
*   **Restoration:** Trivial. The old environment reference is still valid.
*   **Lookup:** O(log n) for a balanced BST.

### Imperative Style
*   **Core Idea:** Mutable, global environment updated destructively. An "undo stack" tracks changes for scope restoration.
*   **Data Structure:** Hash Table with external chaining + auxiliary stack.
*   **Insertion:** New binding is inserted at the **head** of its hash chain. This shadows any previous binding for the same symbol.
    *   Example: Inserting `a ↦ τ2` when `a ↦ τ1` exists results in chain: `hash(a) → <a, τ2> → <a, τ1>`.
*   **Restoration (`pop`):** Removes the head of the chain, revealing the previous binding. Managed via `S_beginScope()`/`S_endScope()` using a marker symbol.
*   **Lookup:** O(1) average case. Looks at the head of the chain first (finds the most recent binding).

## 4. Key Code Components (Tiger Compiler in C)

### 4.1 Hash Table Foundation
*   `struct bucket`: Node for a hash chain (`key`, `binding`, `next`).
*   `hash(char *s)`: Polynomial hash function (65599 multiplier).
*   `insert()`: Adds new bucket to head of chain (enables shadowing).
*   `pop()`: Removes head of chain (undoes last insert for that key).

### 4.2 Symbol Optimization
*   **Problem:** Repeated string hashing/comparison is slow.
*   **Solution: String Interning.** Convert strings to unique `S_symbol` objects.
*   **Benefits:**
    *   Hash key = pointer address (fast).
    *   Equality check = pointer comparison (fast).
    *   Enables fast ordering for BSTs.

### 4.3 Interface (`S_table`, `S_symbol`)
*   `S_symbol S_symbol(string)`: Interns a string, returns unique symbol.
*   `S_enter(S_table, S_symbol, void*)`: Inserts a binding.
*   `S_look(...)`: Looks up a binding.
*   `S_beginScope(S_table)`: Pushes a marker onto the undo stack.
*   `S_endScope(S_table)`: Pops bindings until the marker is found.

### 4.4 Integrated Undo Stack
*   The `Binder` struct (in `TAB_table_`) contains a `prevtop` field.
*   This links bindings in a stack, allowing `S_endScope` to walk back through recent inserts without a separate data structure.
