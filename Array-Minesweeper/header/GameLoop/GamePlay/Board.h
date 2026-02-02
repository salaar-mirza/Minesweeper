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
        Cell* cell;

         float boardWidth = 866.f;
         float boardHeight = 1080.f;
         float boardPosition = 530.f;

        // Board Constants
        static const int numberOfRows = 9;
        static const int numberOfColumns = 9;
        const float horizontalCellPadding = 115.f;
        const float verticalCellPadding = 329.f;

         std::string boardTexturePath = "assets/textures/board.png";
        sf::Texture boardTexture;
        sf::Sprite boardSprite;

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