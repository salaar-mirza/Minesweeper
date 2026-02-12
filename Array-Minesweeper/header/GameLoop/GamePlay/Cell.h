#pragma once
#include "../../header/UI/UIElements/Button/Button.h"
#include <SFML/System/Vector2.hpp>

namespace UIElements
{
    enum class MouseButtonType;
}
namespace Event
{
    class EventPollingManager;
}
namespace sf
{
    class RenderWindow;
}

namespace Gameplay
{
    class  Board;
    
    enum class CellState
    {
        HIDDEN,
        OPEN,
        FLAGGED,
    };

    enum class CellType
    {
        EMPTY,
        ONE,
        TWO,
        THREE,
        FOUR,
        FIVE,
        SIX,
        SEVEN,
        EIGHT,
        MINE,
    };
    
    // Represents a single cell on the Minesweeper board.
    // It manages its own state (hidden, open, flagged), type (mine, empty, number),
    // and handles user interactions by notifying the parent Board.
    class Cell
    {
    public:
        Cell(sf::Vector2i position, float width, float height, Board* board);
        ~Cell();
        
        // Core loop functions
        void update(Event::EventPollingManager& eventManager, sf::RenderWindow& window);
        void render(sf::RenderWindow& window);

        // Cell actions
        void open();
        void toggleFlag();
        void reset();
        bool canOpenCell() const;

        // State management
        CellState getCellState() const;
        void setCellState(CellState state);
        CellType getCellType() const;
        void setCellType(CellType type);
        sf::Vector2i getCellPosition() const;

    private:
        // Constants
        const float cell_top_offset = 274.f;
        const float cell_left_offset = 583.f;
        const int tile_size = 128;
        const int slice_count = 12;
        const std::string cell_texture_path = "assets/textures/cells.jpeg";
        
        // State Variables
        CellState current_cell_state = CellState::HIDDEN;
        CellType cell_type = CellType::EMPTY;
        sf::Vector2i position;

        // Object Pointers
        Board* board;
        UIElements::Button* cell_button;

        // Internal helpers
        void initialize(sf::Vector2i position, float width, float height, Board* board);
        void setCellTexture();
        sf::Vector2f getCellScreenPosition(float width,float height) const;

        // Callback system
        void registerCellButtonCallback();
        void cellButtonCallback(UIElements::MouseButtonType button_type);
    };
}