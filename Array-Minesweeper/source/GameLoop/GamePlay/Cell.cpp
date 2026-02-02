#include "../../header/GameLoop/Gameplay/Cell.h"
#include "../../header/UI/UIElements/Button/Button.h"


namespace Gameplay
{
    Cell::Cell(float width, float height, sf::Vector2i position)
    {
        initialize(width, height, position);
    }

    void Cell::initialize(float width, float height, sf::Vector2i position)
    {
        this->position = position;
        sf::Vector2f cellScreenPosition = getCellScreenPosition();
        cell_button = new UIElements::Button(cell_texture_path, cellScreenPosition, width * slice_count, height); // multiply width by slice count
    }

    sf::Vector2f Cell::getCellScreenPosition() const
    {
        float xScreenPosition = cell_left_offset;
        float yScreenPosition = cell_top_offset;
        return sf::Vector2f(xScreenPosition, yScreenPosition);
    }

    CellState Cell::getCellState() const { return current_cell_state; }

    void Cell::setCellState(CellState state) { current_cell_state = state; }

    CellType Cell::getCellType() const { return cell_type; }

    void Cell::setCellType(CellType type) { cell_type = type; }


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

  
    
    
    void Cell::render(sf::RenderWindow& window) {

        //set cell's texture
        setCellTexture();
        //render the cell button
        if (cell_button) cell_button ->render(window);
    }

}