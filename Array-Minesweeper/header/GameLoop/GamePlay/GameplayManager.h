#pragma once
#include <SFML/Graphics.hpp>

namespace Event
{
    class EventPollingManager;
}
namespace UI
{
    class GameplayUI;
}

namespace Gameplay
{
    class Board; // Forward-declaration

    enum class GameResult
    {
        NONE,
        WON,
        LOST
    };
    
    // Manages the entire gameplay state, including the board, UI, timer, and win/loss conditions.
    // It acts as the central hub for all in-game logic.
    class GameplayManager
    {
    public:
        GameplayManager();
        ~GameplayManager();

        // Core game loop functions
        void update(Event::EventPollingManager& eventManager, sf::RenderWindow& window);
        void render(sf::RenderWindow& window);

        // Public API for game state management
        void restartGame();
        void setGameResult(GameResult gameResult);

    private:
        // Constants
        const float background_alpha = 85.f;
        const float max_level_duration = 150.0f;
        const float game_over_time = 11.0f;
        std::string background_texture_path = "assets/textures/minesweeper_bg.png";
	    
        // State Variables
        float remaining_time;
        GameResult game_result;

        // Object Pointers
        Board* board;
        UI::GameplayUI* gameplay_ui;
        sf::Texture background_texture;
        sf::Sprite background_sprite;
        
        // Initialization methods
        void initialize();
        void initializeVariables();
        void initializeBackground();

        // Game logic methods
        void handleGameplay(Event::EventPollingManager& eventManager, sf::RenderWindow& window);
        void checkGameWin();
        void processGameResult();
        void updateRemainingTime();
        void processTimeOver();
        void gameWon();
        void gameLost();

        bool hasGameEnded() const;
        // Gets the number of mines left to be flagged.
        int getRemainingMinesCount() const;
    };
}