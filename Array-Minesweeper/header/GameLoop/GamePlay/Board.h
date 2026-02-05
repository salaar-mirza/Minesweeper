#pragma once

#include <SFML/Graphics.hpp>
#include <random>
namespace Gameplay
{
    class Cell;
}

namespace Gameplay
{
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

        //Randomization
        std::default_random_engine randomEngine;
        std::random_device randomDevice;
        
        // Board Objects
        sf::Texture boardTexture;
        sf::Sprite boardSprite;
        Cell* cell[numberOfRows][numberOfColumns];
        //Number of Mines
        static const int minesCount = 9;

        void initializeBoardImage();
        void initialize();

        void createBoard();

        float getCellWidthInBoard() const;
        float getCellHeightInBoard() const;

        //Populating the Board
        void populateBoard();
        void populateMines();
        void initializeVariables();

    public:
    		
        Board();
        ~Board();
        
        void render(sf::RenderWindow& window);
    };
}