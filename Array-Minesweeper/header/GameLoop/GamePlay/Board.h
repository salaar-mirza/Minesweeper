#pragma once

#include <SFML/Graphics.hpp>
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

        // Board Objects
        sf::Texture boardTexture;
        sf::Sprite boardSprite;
        Cell* cell;

        void initializeBoardImage();
        void initialize();

        void createBoard();

        float getCellWidthInBoard() const;
        float getCellHeightInBoard() const;

    public:
    		
        Board();
        ~Board();
        
        void render(sf::RenderWindow& window);
    };
}