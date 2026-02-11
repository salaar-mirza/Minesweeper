#include "../../header/GameLoop/Gameplay/Board.h"
#include "../../header/GameLoop/Gameplay/Cell.h"
#include "../../header/Sound/SoundManager.h"
#include "../../header/GameLoop/GamePlay/GameplayManager.h"
#include <iostream>


namespace Gameplay
{
    Board::Board(GameplayManager* gameplayManager)
    { 
        initialize(gameplayManager); 
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

    void Board::onCellButtonClicked(sf::Vector2i cell_position, UIElements::MouseButtonType mouse_button_type) {
        if (mouse_button_type == UIElements::MouseButtonType::LEFT_MOUSE_BUTTON) {
            Sound::SoundManager::PlaySound(Sound::SoundType::BUTTON_CLICK); //play click sound
            openCell(cell_position); // Open the cell when left-clicked
        }
        //Right Click
        else if (mouse_button_type == UIElements::MouseButtonType::RIGHT_MOUSE_BUTTON)
        {
            Sound::SoundManager::PlaySound(Sound::SoundType::FLAG);//play flag sound
            toggleFlag(cell_position);
        }

    }

    void Board::initialize(GameplayManager* gameplayManager)
    {
        initializeVariables(gameplayManager);
        initializeBoardImage();
        createBoard(); 

    }

    void Board::initializeVariables(GameplayManager* gameplay_manager)
    {
        this->gameplay_manager = gameplay_manager;
        randomEngine.seed(randomDevice()); //Function to initialize random engine
        boardState = BoardState::FIRST_CELL;  // Start with first cell state
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

    void Board::createBoard()
    {
        float cell_width = getCellWidthInBoard();
        float cell_height = getCellHeightInBoard();

        for (int row = 0; row < numberOfRows; ++row)
            for (int col = 0; col < numberOfColumns; ++col)
                cell[row][col] = new Cell(sf::Vector2i(col, row),cell_width, cell_height, this); 
    }

    void Board::populateBoard(sf::Vector2i cell_position)
    {
        populateMines(cell_position);
        populateCells();
    }

    void Board::populateMines(sf::Vector2i first_cell_position) 
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

            if (isInvalidMinePosition(first_cell_position, x, y))
                continue;  // Skip first cell's position before placing a mine
    
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

    void Board::openCell(sf::Vector2i cell_position) {
        if (!cell[cell_position.y][cell_position.x]->canOpenCell()) {
            return; // Can't open this cell!
        }
        if (boardState == BoardState::FIRST_CELL) {
            populateBoard(cell_position);    // Place mines after first click
            boardState = BoardState::PLAYING; // Now we can play normally
        }
        processCellType(cell_position);
    }

    void Board::processCellType(sf::Vector2i cell_position) {
        switch (cell[cell_position.y][cell_position.x]->getCellType()) {
        case CellType::EMPTY:
            processEmptyCell(cell_position);
            break;
        case CellType::MINE:
            processMineCell(cell_position);
            break;
        default:
            cell[cell_position.y][cell_position.x]->open();
            break;
        }
    }

    void Board::processEmptyCell(sf::Vector2i cell_position) {
        CellState cell_state = cell[cell_position.y][cell_position.x]->getCellState();

        // Handle the clicked cell
        switch (cell_state) {
        case::Gameplay::CellState::OPEN:
            return;  // Already open, stop here
        default:
            cell[cell_position.y][cell_position.x]->open();
        }

        // Check all 8 neighbors
        for (int row_offset = -1; row_offset <= 1; ++row_offset) {
            for (int col_offset = -1; col_offset <= 1; ++col_offset) {
                //Store neighbor cells position
                sf::Vector2i next_cell_position = sf::Vector2i(cell_position.x + col_offset, cell_position.y + row_offset);

                // Skip current cell and invalid positions
                if ((row_offset == 0 && col_offset == 0) || !isValidCellPosition(next_cell_position)) {
                    continue; 
                }

                //Flagged Cell Case
                CellState next_cell_state = cell[next_cell_position.y][next_cell_position.x]->getCellState();

                if (next_cell_state == CellState::FLAGGED)
                {
                    toggleFlag(next_cell_position);
                }
                
                //Open neighbor cell
                openCell(next_cell_position);  // Open neighbor
            }
        }
    }

    //Handling Mine Cell
    void Board::processMineCell(sf::Vector2i cell_position) {
       gameplay_manager->setGameResult(GameResult::LOST); // Game Over! 
    }
    void Board::revealAllMines() {
        for (int row = 0; row < numberOfRows; row++) {
            for (int col = 0; col < numberOfColumns; col++) {
                if (cell[row][col]->getCellType() == CellType::MINE) {
                    cell[row][col]->setCellState(CellState::OPEN);  // Show the mines
                }
            }
        }
    }
    
    void Board::toggleFlag(sf::Vector2i cell_position) {
        cell[cell_position.y][cell_position.x]->toggleFlag();
        flaggedCells += (cell[cell_position.y][cell_position.x]->getCellState() == CellState::FLAGGED) ? 1 : -1;
    }

    BoardState Board::getBoardState() const 
    { 
        return boardState; 
    }

    void Board::setBoardState(BoardState state) 
    { 
        boardState = state; 
    }

    float Board::getCellWidthInBoard() const
    {
        return (boardWidth - horizontalCellPadding) / numberOfColumns;
    }

    float Board::getCellHeightInBoard() const
    {
        return (boardHeight - verticalCellPadding) / numberOfRows;
    }

    bool Board::isInvalidMinePosition(sf::Vector2i first_cell_position, int x, int y) {
        return (x == first_cell_position.x && y == first_cell_position.y) ||
               cell[y][x]->getCellType() == CellType::MINE;
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


    bool Board::areAllCellsOpen() {
        int total_cells = numberOfRows * numberOfColumns;
        int open_cells = 0;

        for (int row = 0; row < numberOfRows; ++row) {
            for (int col = 0; col < numberOfColumns; ++col) {
                if (cell[row][col]->getCellState() == CellState::OPEN &&
                    cell[row][col]->getCellType() != CellType::MINE) {
                    open_cells++;
                    }
            }
        }

        return open_cells == (total_cells - minesCount);
    }

    void Board::flagAllMines() {
        for (int row = 0; row < numberOfRows; ++row) {
            for (int col = 0; col < numberOfColumns; ++col) {
                if (cell[row][col]->getCellType() == CellType::MINE &&
                    cell[row][col]->getCellState() != CellState::FLAGGED) {
                    cell[row][col]->setCellState(CellState::FLAGGED);
                    }
            }
        }
    }

    int Board::getRemainingMinesCount() const {
        return minesCount - flaggedCells;  // Unflagged mines remaining
    }
   
   
}