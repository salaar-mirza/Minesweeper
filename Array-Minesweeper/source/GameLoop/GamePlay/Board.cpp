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
        for (int row = 0; row < numberOfRows; ++row)
            {
            for (int col = 0; col < numberOfColumns; ++col)
            {
                delete cell[row][col];
            }
        }
        
    }

    void Board::onCellButtonClicked(sf::Vector2i cell_position, UIElements::MouseButtonType mouse_button_type) {
        if (mouse_button_type == UIElements::MouseButtonType::LEFT_MOUSE_BUTTON) {
            // Left-click logic will be added in the next lesson
        } else if (mouse_button_type == UIElements::MouseButtonType::RIGHT_MOUSE_BUTTON) {
            // Right-click logic will be added in the next lesson
        }
    }
    void Board::update(Event::EventPollingManager& eventManager, sf::RenderWindow& window)
    {
        for (int row = 0; row < numberOfRows; ++row)
            for (int col = 0; col < numberOfColumns; ++col)
                cell[row][col]->update(eventManager, window);
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

    void Board::initialize()
    {
        initializeBoardImage();
        initializeVariables();
        createBoard(); //Call Create Board method:
        populateBoard();

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

    void Board::initializeVariables()
    {
        randomEngine.seed(randomDevice()); //Function to initialize random engine
    }

    void Board::createBoard()
    {
        float cell_width = getCellWidthInBoard();
        float cell_height = getCellHeightInBoard();

        for (int row = 0; row < numberOfRows; ++row)
            for (int col = 0; col < numberOfColumns; ++col)
                cell[row][col] = new Cell(sf::Vector2i(col, row),cell_width, cell_height, this); //pass the board as a parameter
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
        populateCells();
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
            if (cell[y][x]->getCellType() != CellType::MINE) {
                //Step 4
                cell[y][x]->setCellType(CellType::MINE);
                ++mines_placed;
            }
        }
   
    }

    void Board::populateCells()
    {
        for (int row = 0; row < numberOfRows; ++row)
            for (int col = 0; col < numberOfColumns; ++col)
                if (cell[row][col]->getCellType() != CellType::MINE)
                {
                    int mines_around = countMinesAround(sf::Vector2i(col, row));
                    cell[row][col]->setCellType(static_cast<CellType>(mines_around));
                }
    }

    int Board::countMinesAround(sf::Vector2i cell_position) { // cell_position is (x,y) -> (col, row)
        // local variable to keep track of cell value
        int mines_around = 0;
        
        for (int row_offset = -1; row_offset <= 1; ++row_offset) {
            for (int col_offset = -1; col_offset <= 1; ++col_offset) {
                // Validate cell's postion and check current cell
                if ((row_offset == 0 && col_offset == 0) ||
                        !isValidCellPosition(sf::Vector2i(cell_position.x + col_offset, cell_position.y + row_offset)))
                    continue;

                //Check Mines
                if (cell[cell_position.y + row_offset][cell_position.x + col_offset]->getCellType() ==
                CellType::MINE)
                    mines_around++;
            }
            
        }
        return mines_around;
    }

    bool Board::isValidCellPosition(sf::Vector2i cell_position) // cell_position is (x,y) -> (col, row)
    {
        return (cell_position.x >= 0 && cell_position.y >= 0 &&
            cell_position.x < numberOfColumns && cell_position.y < numberOfRows);
    }
    
}