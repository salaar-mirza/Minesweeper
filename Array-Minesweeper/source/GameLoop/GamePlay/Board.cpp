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
        for (int col = 0; col < numberOfColumns; ++col)
            {
            for (int row = 0; row < numberOfRows; ++row)
            {
                delete cell[row][col];
            }
        }
        
    }

    void Board::initialize()
    {
        initializeBoardImage();
        initializeVariables();
        createBoard(); //Call Create Board method:
        populateBoard();

    }

    void Board::initializeVariables()
    {
        randomEngine.seed(randomDevice()); //Function to initialize random engine
    }
    void Board::createBoard()
    {
        float cell_width = getCellWidthInBoard();
        float cell_height = getCellHeightInBoard();
        //create cells for the cell[][] array
        for (int row = 0; row < numberOfRows; ++row)
        {
            for (int col = 0; col < numberOfColumns; ++col)
            {
                cell[row][col] = new Cell(sf::Vector2i(row, col),cell_width, cell_height);
            }
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

    void Board::populateBoard()
    {
        populateMines();
    }


    void Board::populateMines() 
    {
        //Step 1
        std::uniform_int_distribution<int> x_dist(0, numberOfColumns - 1);
        std::uniform_int_distribution<int> y_dist(0, numberOfRows - 1); 		
		
        //Step 2
        int mines_placed = 0;
        while (mines_placed < minesCount)
        {
            int x = x_dist(randomEngine);
            int y = y_dist(randomEngine);
    
            //Step 3
            if (cell[x][y]->getCellType() != CellType::MINE) {
                //Step 4
                cell[x][y]->setCellType(CellType::MINE);
                ++mines_placed;
            }
        }
   
    }

    

    void Board::render(sf::RenderWindow& window)
    {
        window.draw(boardSprite);
    
        for (int row = 0; row < numberOfRows; ++row)
        {
            for (int col = 0; col < numberOfColumns; ++col)
                cell[row][col]->render(window);
        }
     }
    
}