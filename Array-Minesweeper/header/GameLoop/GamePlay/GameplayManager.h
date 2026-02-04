#pragma once
#include "../../header/GameLoop/Gameplay/Board.h"
#include <SFML/Graphics.hpp>

namespace Gameplay
{
    class GameplayManager
    {
    private:
        const float background_alpha = 85.f;
        std::string background_texture_path = "assets/textures/minesweeper_bg.png";
	    
        sf::Texture background_texture;
        sf::Sprite background_sprite;

        Board* board;
        
        void initialize();
        void initializeVariables();
        void initializeBackground();


    public:
        GameplayManager();
        ~GameplayManager();

        void render(sf::RenderWindow& window);
    };
}