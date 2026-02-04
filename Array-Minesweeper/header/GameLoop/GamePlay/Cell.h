#pragma once

#include <SFML/Graphics.hpp>


namespace UIElements
{
    class Button;
}

namespace Event
{
    class EventPollingManager;
}


namespace Gameplay
{

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
    
    
    class Cell
    {
    private:
        // Constants
        const float cell_top_offset = 274.f;
        const float cell_left_offset = 583.f;
        const int tile_size = 128;
        const int slice_count = 12;
        const std::string cell_texture_path = "assets/textures/cells.jpeg";
        
        // Cell data members
        CellState current_cell_state = CellState::HIDDEN;
        CellType cell_type = CellType::EMPTY;
        sf::Vector2i position;

        UIElements::Button* cell_button;

        void initialize(sf::Vector2i position, float width, float height);

        sf::Vector2f getCellScreenPosition(float width,float height) const;

    public:
        Cell(sf::Vector2i position, float width, float height);
        ~Cell() = default;

        //Getters, Setters
        CellState getCellState() const;
        void setCellState(CellState state);
        CellType getCellType() const;
        void setCellType(CellType type);
        
        void setCellTexture();
        void render(sf::RenderWindow& window);
    };
}