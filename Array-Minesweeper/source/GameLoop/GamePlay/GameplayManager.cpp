#include "../../header/GameLoop/Gameplay/GameplayManager.h"
#include <iostream>

namespace Gameplay
{
    GameplayManager::GameplayManager()
    {
        initialize();
    }

    GameplayManager::~GameplayManager()
    {
        delete board;
    }

    void GameplayManager::initialize()
    {
        initializeBackground();
        initializeVariables();
    }

    void GameplayManager::initializeVariables()
    {
        board = new Board();
    }

    void GameplayManager::initializeBackground()
    {
        if (!background_texture.loadFromFile(background_texture_path)) {
            std::cerr << "Failed to load background texture!\n";
        }
        background_sprite.setTexture(background_texture);
        background_sprite.setColor(sf::Color(255, 255, 255, background_alpha));
    }

    void GameplayManager::render(sf::RenderWindow& window)
    {
        window.draw(background_sprite);
        board->render(window);
        
    }
}