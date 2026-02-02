#include "../../header/GameLoop/Gameplay/Cell.h"
#include "../../header/UI/UIElements/Button/Button.h"


namespace Gameplay
{
    Cell::Cell(float width, float height, sf::Vector2i position)
    {
        initialize(width, height, position);
    }

    /*void Cell::initialize(float width, float height, sf::Vector2i position)
    {
        this->position = position;
        sf::Vector2f float_position(static_cast<float>(position.x), static_cast<float>(position.y));  //Convert int to float
        cell_button = new UIElements::Button(cell_texture_path, float_position, width, height);
    }*/

    void Cell::initialize(float width, float height, sf::Vector2i position)
    {
        this->position = position;
        sf::Vector2f float_position(static_cast<float>(position.x), static_cast<float>(position.y));     
        cell_button = new UIElements::Button(cell_texture_path, float_position, width * slice_count, height); // multiply width by slice count
    }

    CellState Cell::getCellState() const { return current_cell_state; }

    void Cell::setCellState(CellState state) { current_cell_state = state; }

    CellType Cell::getCellType() const { return cell_type; }

    void Cell::setCellType(CellType type) { cell_type = type; }


    void Cell::setCellTexture()
    {
        switch (current_cell_state)
        {
        case CellState::OPEN:
            switch (cell_type)
            {
        case CellType::EMPTY:
                cell_button->setTextureRect(sf::IntRect(0 * tile_size, 0, tile_size, tile_size));
                break;
        case CellType::ONE:
                cell_button->setTextureRect(sf::IntRect(1 * tile_size, 0, tile_size, tile_size));
                break;
        case CellType::TWO:
                cell_button->setTextureRect(sf::IntRect(2 * tile_size, 0, tile_size, tile_size));
                break;
        case CellType::THREE:
                cell_button->setTextureRect(sf::IntRect(3 * tile_size, 0, tile_size, tile_size));
                break;
        case CellType::FOUR:
                cell_button->setTextureRect(sf::IntRect(4 * tile_size, 0, tile_size, tile_size));
                break;
        case CellType::FIVE:
                cell_button->setTextureRect(sf::IntRect(5 * tile_size, 0, tile_size, tile_size));
                break;
        case CellType::SIX:
                cell_button->setTextureRect(sf::IntRect(6 * tile_size, 0, tile_size, tile_size));
                break;
        case CellType::SEVEN:
                cell_button->setTextureRect(sf::IntRect(7 * tile_size, 0, tile_size, tile_size));
                break;
        case CellType::EIGHT:
                cell_button->setTextureRect(sf::IntRect(8 * tile_size, 0, tile_size, tile_size));
                break;
        case CellType::MINE:
                cell_button->setTextureRect(sf::IntRect(9 * tile_size, 0, tile_size, tile_size));
                break;
            }
            break;

        case CellState::HIDDEN:
            cell_button->setTextureRect(sf::IntRect(10 * tile_size, 0, tile_size, tile_size));
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