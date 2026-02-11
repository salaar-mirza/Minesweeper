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

    void run();
    static void setGameState(GameState state_to_set);
};