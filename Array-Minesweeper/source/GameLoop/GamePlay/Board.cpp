#include "../../header/GameLoop/Gameplay/Board.h"
#include "../../header/GameLoop/Gameplay/Cell.h"
#include <iostream>


namespace Gameplay
{
    Board::Board()
    {
        initialize();

    }

    Board::~Board()
    {
        for (int col = 0; col < numberOfColumns; ++col) {
            delete cell[col];
        }
        
    }

    void Board::initialize()
    {
        initializeBoardImage();
        createBoard(); //Call Create Board method:

    }

  
    void Board::createBoard()
    {
        float cell_width = getCellWidthInBoard();
        float cell_height = getCellHeightInBoard();
        //Create a cell for each array index
        for (int col = 0; col < numberOfColumns; ++col) {
            cell[col] = new Cell(sf::Vector2i(col, 0),cell_width, cell_height);
        }
    }

    
    void Board::initializeBoardImage() {
        if (!boardTexture.loadFromFile(boardTexturePath)) {
            std::cout << "Failed to load board texture!\n";
            return;
        }
    
        boardSprite.setTexture(boardTexture);
        boardSprite.setPosition(boardPosition, 0);
        boardSprite.setScale(boardWidth / boardTexture.getSize().x,
                            boardHeight / boardTexture.getSize().y);
    }

    float Board::getCellWidthInBoard() const
    {
        return (boardWidth - horizontalCellPadding) / numberOfColumns;
    }

    float Board::getCellHeightInBoard() const
    {
        return (boardHeight - verticalCellPadding) / numberOfRows;
    }


    void Board::render(sf::RenderWindow& window)
    {
        window.draw(boardSprite);
        //render array's elements one by one
        for (int col = 0; col < numberOfColumns; ++col) {
            cell[col]->render(window);
        } 
    }


    
}