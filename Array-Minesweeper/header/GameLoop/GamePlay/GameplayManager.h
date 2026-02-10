#pragma once
#include "../../header/GameLoop/Gameplay/Board.h"
#include <SFML/Graphics.hpp>

namespace Event
{
    class EventPollingManager;
}

namespace Gameplay
{
    enum class GameResult
    {
        NONE,
        WON,
        LOST
    };
    
    class GameplayManager
    {
    private:
        const float background_alpha = 85.f;
        std::string background_texture_path = "assets/textures/minesweeper_bg.png";
	    
        sf::Texture background_texture;
        sf::Sprite background_sprite;
        
        Board* board;

        GameResult game_result;
        
        void initialize();
        void initializeVariables();
        void initializeBackground();

        bool hasGameEnded();
    public:
        GameplayManager();
        ~GameplayManager();

        void update(Event::EventPollingManager& eventManager, sf::RenderWindow& window);
        void setGameResult(GameResult gameResult);

        void render(sf::RenderWindow& window);
    };
}