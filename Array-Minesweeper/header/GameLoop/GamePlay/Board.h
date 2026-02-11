#pragma once

#include <SFML/Graphics.hpp>
#include <random>

#include "../../UI/UIElements/Button/Button.h"

namespace Gameplay
{
    class Cell;
    class GameplayManager; 
}
namespace Event
{
    class EventPollingManager;
}

namespace Gameplay
{
    enum class BoardState
    {
        FIRST_CELL,
        PLAYING,
        COMPLETED,
    };
		
    
    class Board
    {
    private:
        // Board Constants
        static const int numberOfRows = 9;
        static const int numberOfColumns = 9;
        const float horizontalCellPadding = 115.f;
        const float verticalCellPadding = 329.f;

        // Board Data
        float boardWidth = 866.f;
        float boardHeight = 1080.f;
        float boardPosition = 530.f;
        std::string boardTexturePath = "assets/textures/board.png";

        int flaggedCells;

        //Randomization
        std::default_random_engine randomEngine;
        std::random_device randomDevice;

        BoardState boardState;
        // Board Objects
        sf::Texture boardTexture;
        sf::Sprite boardSprite;
        Cell* cell[numberOfRows][numberOfColumns];
        //Number of Mines
        static const int minesCount = 9;

        GameplayManager* gameplay_manager;
      

        void initializeBoardImage();
        void initialize(GameplayManager* gameplay_manager);
        void initializeVariables(GameplayManager* gameplay_manager);

       
        void createBoard();

        float getCellWidthInBoard() const;
        float getCellHeightInBoard() const;

        //Populating the Board
        void populateBoard(sf::Vector2i cell_position);
        void populateMines(sf::Vector2i first_cell_position);

        int countMinesAround(sf::Vector2i cell_position);
        void populateCells();
        bool isValidCellPosition(sf::Vector2i cell_position);

        void openCell(sf::Vector2i cell_position);

        void toggleFlag(sf::Vector2i cell_position);

        //Cell Type
        void processCellType(sf::Vector2i cell_position);
        //Empty Cells
        void processEmptyCell(sf::Vector2i cell_position);

        void processMineCell(sf::Vector2i cell_position);
        
        bool isInvalidMinePosition(sf::Vector2i first_cell_position, int x, int y);

      
    public:
    		
        Board(GameplayManager* gameplayManager);
        ~Board();
        void onCellButtonClicked(sf::Vector2i cell_position,UIElements:: MouseButtonType mouse_button_type);
        void revealAllMines();

        BoardState getBoardState() const;
        void setBoardState(BoardState state);

        void update(Event::EventPollingManager& eventManager, sf::RenderWindow& window);
        bool areAllCellsOpen();
        void flagAllMines();
        
        void render(sf::RenderWindow& window);
    };
}