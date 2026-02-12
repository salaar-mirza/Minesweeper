#include "../../header/GameLoop/GameLoop.h"
#include <iostream>
#include "../../header/Event/EventPollingManager.h"
#include  "../../header/GameLoop/GamePlay/GameplayManager.h"
#include "../../header/GameWindow/GameWindowManager.h"
#include "../../header/Sound/SoundManager.h"
#include "../../header/Time/TimeManager.h"
#include "../../header/UI/MainMenu/MainMenuManager.h"
#include "../../header/UI/SplashScreen/SplashScreenManager.h"

GameState GameLoop::current_state = GameState::SPLASH_SCREEN;

GameLoop::GameLoop() { initialize(); }

GameLoop::~GameLoop()
{
    delete window_manager;
    delete event_manager;
    delete splash_screen_manager;
    delete main_menu_manager; 
    delete gameplay_manager;
}

void GameLoop::run()
{
    while (window_manager->isGameWindowOpen())
    {
        event_manager->processEvents();
        update();
        render();
    }
}

void GameLoop::setGameState(GameState state_to_set) { GameLoop::current_state = state_to_set; }

void GameLoop::initialize()
{
    // Create Managers:
    window_manager = new GameWindow::GameWindowManager();
    game_window = window_manager->getGameWindow();
    event_manager = new Event::EventPollingManager(game_window);
    splash_screen_manager = new UI::SplashScreenManager(game_window);
    main_menu_manager = new UI::MainMenuManager(game_window);
    gameplay_manager = new Gameplay::GameplayManager();
    
    // Initialize Sounds:
    Sound::SoundManager::Initialize();
    Sound::SoundManager::PlayBackgroundMusic();

    // Initialize Time:
    Time::TimeManager::initialize();
}

// Updates the game logic. This is called once per frame.
// It updates the time, processes events, and then calls the update function
// of the manager corresponding to the current game state.
void GameLoop::update()
{
    Time::TimeManager::update();
    event_manager->update();
    window_manager->update();

    switch (current_state)
    {
    case GameState::SPLASH_SCREEN:
        splash_screen_manager->update();
        break;
    case GameState::MAIN_MENU:
        main_menu_manager->update(*event_manager); 
        break;
    case GameState::GAMEPLAY:
        gameplay_manager->update(*event_manager, *game_window); 
        break;
    case GameState::EXIT:
        game_window->close();
        break;
    }
}

// Renders the game. This is called once per frame.
// It clears the window, and then calls the render function
// of the manager corresponding to the current game state.
void GameLoop::render()
{
    game_window->clear();
    window_manager->render();

    switch (current_state)
    {
    case GameState::SPLASH_SCREEN:
        splash_screen_manager->render();
        break;
    case GameState::MAIN_MENU:
        main_menu_manager->render(); 
        break;
    case GameState::GAMEPLAY:
        gameplay_manager->render(*game_window);
        break;
    case GameState::EXIT:
        std::cout<<"Exited the Game loop Successfully\n"; 
        break;
    }
    
    game_window->display();
}