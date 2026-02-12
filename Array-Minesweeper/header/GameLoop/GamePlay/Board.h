#pragma once

#include <SFML/Graphics.hpp>
#include <random>

#include "../../UI/UIElements/Button/Button.h"

namespace Gameplay
{
    class GameplayManager; 
}
namespace Event
{
    class EventPollingManager;
}

namespace Gameplay
{
    class Cell;
    enum class BoardState
    {
        FIRST_CELL,
        PLAYING,
        COMPLETED,
    };
		
    // Manages the game board itself, including the grid of cells, mine placement,
    // and the core win/loss logic based on cell states.
    class Board
    {
    public:
        Board(GameplayManager* gameplayManager);
        ~Board();

        // Core loop functions
        void update(Event::EventPollingManager& eventManager, sf::RenderWindow& window);
        void render(sf::RenderWindow& window);

        // Public API for game interactions
        void onCellButtonClicked(sf::Vector2i cell_position, UIElements::MouseButtonType mouse_button_type);
        void reset();

        // State query methods
        BoardState getBoardState() const;
        bool areAllCellsOpen();
        int getRemainingMinesCount() const;

        // State modification methods
        void setBoardState(BoardState state);
        void flagAllMines();
        void revealAllMines();

    private:
        // Constants
        static const int numberOfRows = 9;
        static const int numberOfColumns = 9;
        static const int minesCount = 9;
        const float horizontalCellPadding = 115.f;
        const float verticalCellPadding = 329.f;

        // Configuration Data
        float boardWidth = 866.f;
        float boardHeight = 1080.f;
        float boardPosition = 530.f;
        std::string boardTexturePath = "assets/textures/board.png";

        // State Variables
        int flaggedCells;
        BoardState boardState;

        // Object Pointers & Members
        GameplayManager* gameplay_manager;
        Cell* cell[numberOfRows][numberOfColumns];
        sf::Texture boardTexture;
        sf::Sprite boardSprite;

        // Utility Members
        std::default_random_engine randomEngine;
        std::random_device randomDevice;

        // Initialization
        void initialize(GameplayManager* gameplay_manager);
        void initializeVariables(GameplayManager* gameplay_manager);
        void initializeBoardImage();
        void createBoard();

        // Board Population
        void populateBoard(sf::Vector2i cell_position);
        void populateMines(sf::Vector2i first_cell_position);
        void populateCells();
        int countMinesAround(sf::Vector2i cell_position) const;

        // Cell Interaction Logic
        void openCell(sf::Vector2i cell_position);
        void toggleFlag(sf::Vector2i cell_position);
        void processCellType(sf::Vector2i cell_position);
        void processEmptyCell(sf::Vector2i cell_position);
        void processMineCell(sf::Vector2i cell_position);

        // Helper Methods
        float getCellWidthInBoard() const;
        float getCellHeightInBoard() const;
        bool isValidCellPosition(sf::Vector2i cell_position) const;
        bool isInvalidMinePosition(sf::Vector2i first_cell_position, int x, int y) const;
    };
}