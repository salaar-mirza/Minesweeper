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
        
        const float cell_top_offset = 274.f;
        const float cell_left_offset = 583.f;
        // Cell data members
        CellState current_cell_state = CellState::HIDDEN;
        CellType cell_type = CellType::EMPTY;
        
        sf::Vector2i position;

        const int tile_size = 128;
        const int slice_count = 12;
        const std::string cell_texture_path = "assets/textures/cells.jpeg";
        
        UIElements::Button* cell_button;

        void initialize(float width, float height, sf::Vector2i position);

        sf::Vector2f getCellScreenPosition(float width,float height) const;

    public:
        Cell(float width, float height, sf::Vector2i position);
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