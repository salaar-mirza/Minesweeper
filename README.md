# Minesweeper

A robust, object-oriented implementation of the classic Minesweeper game, built from scratch using **C++** and the **SFML** library.

This project demonstrates a custom game engine architecture with a focus on **separation of concerns**, **memory management**, and **scalable design patterns**.

## Features

*   **Classic Gameplay**: Authentic Minesweeper logic with recursive flood-fill for empty cells.
*   **Procedural Generation**: Mines are randomized every game using `std::random_device`.
*   **Safe Start**: The first click is guaranteed to be safe (mines are generated *after* the first interaction).
*   **Game Loop Engine**: Custom-built game loop handling Input, Update, and Render cycles.
*   **Component Architecture**: Modular design using specialized Managers (Gameplay, UI, Event, Sound, Time).
*   **Audio System**: Sound effects for clicks, flags, explosions, and victory, plus background music.
*   **UI & HUD**: Real-time mine counter, timer, and interactive restart button.
*   **Win/Loss States**: Automatic detection of victory or defeat conditions.

<p align="center">
  <img src="Screenshorts\MineSweeper1.png" width="45%" />
  <img src="Screenshorts\MineSweeper2.png" width="45%" />
</p>
<p align="center">
  <img src="Screenshorts\MineSweeper3.png" width="45%" />
</p>

## Architecture & Design

The project follows a **Composition-based Architecture**, deliberately avoiding global Singletons in favor of **Dependency Injection**. The `GameLoop` acts as the central orchestrator, managing the lifecycle of specialized systems.

### Key Design Decisions
1.  **No Singletons**: We avoid the Singleton pattern to prevent hidden dependencies and global state pollution. Instead, dependencies (like `Board` or `SoundManager`) are injected via constructors.
2.  **Row-Major Memory Layout**: The 2D grid is stored as `cell[row][col]` to align with C++ memory standards and optimize CPU cache usage.
3.  **Separation of Concerns**:
    *   **Model**: `Board` and `Cell` handle the data and logic.
    *   **View**: `GameplayUI` and `Button` handle the rendering.
    *   **Controller**: `GameplayManager` and `EventPollingManager` handle the flow and input.

### UML Class Diagram
The following diagram illustrates the high-level architecture of the game engine:

<p align="center">
  <img src="UML_Diagram\diagram6.1.png" alt="Minesweeper UML Diagram" />
</p>


## Core Algorithms

This project implements several key algorithms for grid management and gameplay:
*   **Procedural Mine Placement** (Randomized with exclusion zones).
*   **Neighbor Calculation** (3x3 Kernel convolution).
*   **Recursive Flood Fill** (Depth-First Search for opening empty areas).

**Read the detailed Core Logic Documentation here**

## Controls

*   **Left Click**: Reveal a cell.
*   **Right Click**: Flag/Unflag a cell to mark a suspected mine.
*   **Restart Button**: Click the Globe face button to reset the board and timer.

## Project Structure

```text
📂 Source
├── 📂 Event           # Input pollinghandling and event  
├── 📂 GameLoop        # The main engine loop and state management
├    └── 📂 Gameplay   # Core game logic (Board, Cell, Rules)
├── 📂 GameWindow      # SFML Window creation and configuration
├── 📂 Sound           # Audio management
├── 📂 Time            # Delta time calculation
└── 📂  UI              # HUD, Buttons, and Menus
```

## Getting Started

### Prerequisites
*   C++ Compiler (C++17 recommended)
*   SFML Library (2.5+)

### Build Instructions
1.  Ensure SFML headers and libraries are linked in your project settings.
2.  Place the `assets` folder in the same directory as your executable.
3.  Build and Run!