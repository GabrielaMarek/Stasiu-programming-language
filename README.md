# Stasiu Programming Language
The code is executable in most cases, you must create a file named "example.stas" (or whatever you want) and run the shell.py. In the terminal run your file:
```stasiu
run("example1.stas")
```


Stasiu is a dynamically-typed, interpreted programming language designed for simplicity and educational purposes. It features a clean syntax with influences from Python and pseudocode conventions, making it ideal for beginners learning programming concepts.

## Features

### Variables & Data Types
- Numbers (integers/floats)
- Strings
- Lists
- Boolean logic (via 1/0)

### Control Flow
- Conditional statements (`when`, `otherwise when`, `in any other case`)
- Loops:
  - Range-based (`repeat from...to`)
  - Count-based (`repeat X times`)
  - While loops (`while...then`)

### Operations
- Arithmetic (`+`, `-`, `*`, `/`, `^`)
- Comparison (`<`, `>`, `==`, `!=`, `<=`, `>=`)
- Logical (`and`, `or`, `not`)
- List indexing

### I/O Operations
- Display output to console
- Variable creation/assignment

### Error Handling
- Syntax highlighting in error messages
- Detailed runtime errors
- Traceback support

---

## Installation
Ensure Python 3.10+ is installed, then clone the repository:

```bash
git clone https://github.com/GabrielaMarek/Stasiu-programming-language
cd Stasiu-programming-language
```

---

## Quick Start

### Hello World
```stasiu
display: "Hello World!"
```

### Variables
```stasiu
x = 10
display: x + 5  # Outputs 15
```

### Conditional Logic
```stasiu
ask: "How old are you? " then save to age

when (age >= 0 and age < 13) then:
    display: "You are a child"
otherwise when (age >= 13 and age < 20) then:
    display: "You are a teenager"
in any other case:
    display: "You are: " + age + " years old."

```

### Loops
#### Range-based:
```stasiu
repeat i from 1 to 5 step 1:
    display: i
# Outputs 1 2 3 4 5
```

#### Count-based:
```stasiu
repeat 3 times:
    display: "Hello"
# Outputs Hello 3 times
```

### Lists
```stasiu
create fruits = ["apple", "banana", "cherry"]
display: fruits[1]  # Outputs "banana"

create numbers = [1, 2, 3]
add 4 to numbers
remove 2 from numbers
remove index 0 from numbers
```

---

## Full Syntax Guide

### Variables
```stasiu
name = "Stasiu"
age = 20
weight = 3.14
```

### Operators
#### Arithmetic:
```stasiu
5 + 3   # 8
10 - 2  # 8
4 * 2   # 8
16 / 2  # 8
2 ^ 3   # 8
```

#### Comparison:
```stasiu
5 == 5  # 1 (true)
3 != 2  # 1
5 > 3   # 1
2 < 10  # 1
```

#### Logical:
```stasiu
1 and 0  # 0
1 or 0   # 1
not 1    # 0
```

---

## Error Handling
Stasiu provides detailed error messages:

```
Runtime Error: Division by zero
File example.st, line 5
  3 | create x = 0
  4 | create y = 5
>5 | display: y / x
           ^
```

---

## Architecture Overview
- **Lexer**: Tokenizes source code
- **Parser**: Builds Abstract Syntax Tree (AST)
- **Interpreter**: Executes AST with runtime context
- **Error Handling**: Detailed error reporting system
- **Key components**:
  - Symbol tables for variable management
  - Context-aware execution for scoping
  - Type coercion rules
  - Garbage collection via Python's reference counting

---

## Roadmap
- Add function support ✅
- Implement modules system 🛠️
- Add dictionary/hashmap type 🛠️
- Develop debugger 🛠️
- Create package manager 🛠️
  
---



