#include "../../header/GameLoop/Gameplay/Cell.h"
#include "../../header/UI/UIElements/Button/Button.h"
#include "../../header/GameLoop/GamePlay/Board.h"
#include "../../header/Event/EventPollingManager.h"


namespace Gameplay
{
    Cell::Cell(sf::Vector2i position, float width, float height, Board* board)
    {
        initialize(position, width, height, board);
    }

    Cell::~Cell()
    {
        delete cell_button;
    }

    void Cell::update(Event::EventPollingManager& eventManager, sf::RenderWindow& window)
    {
        if (cell_button)
            cell_button->handleButtonInteractions(eventManager, window);
    }

    void Cell::render(sf::RenderWindow& window) {
        if (cell_button)
        {
            setCellTexture();
            cell_button->render(window);
        }
    }

    void Cell::open() {
        setCellState(CellState::OPEN); // Change state to OPEN
    }

    void Cell::toggleFlag() {
        if (current_cell_state == CellState::HIDDEN) {
            setCellState(CellState::FLAGGED);
        } else if (current_cell_state == CellState::FLAGGED) {
            setCellState(CellState::HIDDEN);
        }
    }

    void Cell::reset() {
        current_cell_state = CellState::HIDDEN;  // Back to hidden
        cell_type = CellType::EMPTY;            // Back to empty
    }

    bool Cell::canOpenCell() const { return current_cell_state == CellState::HIDDEN; }

    CellState Cell::getCellState() const { return current_cell_state; }

    void Cell::setCellState(CellState state) { current_cell_state = state; }

    CellType Cell::getCellType() const { return cell_type; }

    void Cell::setCellType(CellType type) { cell_type = type; }

    sf::Vector2i Cell::getCellPosition() const { return position; }

    void Cell::initialize(sf::Vector2i position, float width, float height, Board* board)
    {
        this->position = position; // will be used in the future content
        this->board = board;
        sf::Vector2f cellScreenPosition = getCellScreenPosition(width, height);
        cell_button = new UIElements::Button(cell_texture_path, cellScreenPosition, width * slice_count, height);
        current_cell_state = CellState::HIDDEN;
        
        registerCellButtonCallback();
    }
    
    sf::Vector2f Cell::getCellScreenPosition(float width, float height) const
    {
        float xScreenPosition = cell_left_offset + position.x * width;
        float yScreenPosition = cell_top_offset + position.y * height;
        return sf::Vector2f(xScreenPosition, yScreenPosition);
    }

    void Cell::setCellTexture()
    {
        int index = static_cast<int>(cell_type);

        switch (current_cell_state)
        {
        case CellState::HIDDEN:
            cell_button->setTextureRect(sf::IntRect(10 * tile_size, 0, tile_size, tile_size));
            break;

        case CellState::OPEN:
            cell_button->setTextureRect(sf::IntRect(index * tile_size, 0, tile_size, tile_size));
            break;

        case CellState::FLAGGED:
            cell_button->setTextureRect(sf::IntRect(11 * tile_size, 0, tile_size, tile_size));
            break;
        }
    }

    void Cell::registerCellButtonCallback() {
        cell_button->registerCallbackFunction([this](UIElements::MouseButtonType button_type) {
            cellButtonCallback(button_type);  // Call Cell's own callback logic
        });
    }

    void Cell::cellButtonCallback(UIElements::MouseButtonType button_type) {
        board->onCellButtonClicked(getCellPosition(), button_type);
    }
}