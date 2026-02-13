#pragma once

//Forward Declarations since we used Pointer Types.
namespace sf
{
    class RenderWindow;
}
namespace  GameWindow
{
    class GameWindowManager;
}
namespace Event
{
    class EventPollingManager;
}
namespace UI
{
    class SplashScreenManager;
}

namespace Gameplay
{
    class GameplayManager;
    
}
namespace UI
{
    class MainMenuManager;
}


enum class GameState
{
    SPLASH_SCREEN,
    MAIN_MENU,
    GAMEPLAY,
    EXIT
};

 
// The main orchestrator of the game. This class owns all the major managers
// and runs the main game loop, delegating tasks for initialization,
// updates, and rendering based on the current game state.
class GameLoop {
private:
    static GameState current_state;

    sf::RenderWindow* game_window;
    GameWindow::GameWindowManager* window_manager;
    Event::EventPollingManager* event_manager;
    UI::SplashScreenManager* splash_screen_manager;
    Gameplay::GameplayManager* gameplay_manager;
    UI::MainMenuManager* main_menu_manager;

    void initialize();
    void update();
    void render();

public:
    GameLoop();
    ~GameLoop();

    // Main entry point for the game.
    void run();
    // Static method to allow any part of the game to change the global state.
    static void setGameState(GameState state_to_set);
};