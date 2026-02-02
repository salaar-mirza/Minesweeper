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
        delete cell;
    }

    void Board::initialize()
    {
        initializeBoardImage();
        createBoard(); //Call Create Board method:

    }

    void Board::createBoard() {
        cell = new Gameplay::Cell(83, 83, sf::Vector2i(0, 0));
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

    void Board::render(sf::RenderWindow& window)
    {
        window.draw(boardSprite);
        cell->render(window);   //Render the cell   
    }


    
}