#pragma once
#include "../../header/GameLoop/Gameplay/Board.h"
#include <SFML/Graphics.hpp>

namespace Event
{
    class EventPollingManager;
}

namespace Time
{
    class TimeManager;
}

namespace UI
{
    class GameplayUI;
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

        //Timer
        const float max_level_duration = 150.0f;
        const float game_over_time = 11.0f;
        float remaining_time;

        
        Board* board;
        UI::GameplayUI* gameplay_ui;


        GameResult game_result;
        
        void initialize();
        void initializeVariables();
        void initializeBackground();

        bool hasGameEnded();

        void updateRemainingTime();
        void processTimeOver();
        void gameWon();
        void gameLost();

        void handleGameplay(Event::EventPollingManager& eventManager, sf::RenderWindow& window);
        int getRemainingMinesCount() const;

    public:
        GameplayManager();
        ~GameplayManager();

        void update(Event::EventPollingManager& eventManager, sf::RenderWindow& window);
        void setGameResult(GameResult gameResult);
        void checkGameWin();
        void processGameResult();

        void render(sf::RenderWindow& window);
    };
}