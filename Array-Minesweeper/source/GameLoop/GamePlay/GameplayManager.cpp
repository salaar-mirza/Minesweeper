#include "../../header/GameLoop/Gameplay/GameplayManager.h"
#include "../../header/Sound/SoundManager.h"
#include "../../header/Time/TimeManager.h"
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

    void GameplayManager::update(Event::EventPollingManager& eventManager, sf::RenderWindow& window) {
        if (!hasGameEnded())
            handleGameplay(eventManager, window);
        else if (board->getBoardState() != BoardState::COMPLETED)
            processGameResult();  // Handle win/loss
    }
   
    void GameplayManager::handleGameplay(Event::EventPollingManager& eventManager, sf::RenderWindow& window) {
        updateRemainingTime();              // Update timer first
        board->update(eventManager, window); // Then update board
        checkGameWin();  // See if player has won
    }

    void GameplayManager::checkGameWin() {
        if (board->areAllCellsOpen()) {
            game_result = GameResult::WON;  // Victory!
        }
    }

    void GameplayManager::processGameResult() {
        switch (game_result) {
        case GameResult::WON:
            gameWon();    // Victory! 
            break;
        case GameResult::LOST:
            gameLost();   // Game Over! 
            break;
        default:
            break;
        }
    }

    void GameplayManager::gameWon() {
        Sound::SoundManager::PlaySound(Sound::SoundType::GAME_WON);  // Play victory sound
        board->flagAllMines();  // Show all mines
        board->setBoardState(BoardState::COMPLETED);  // Stop the game
    }
    void GameplayManager::gameLost() {
        Sound::SoundManager::PlaySound(Sound::SoundType::EXPLOSION);  // Boom!
        board->setBoardState(BoardState::COMPLETED);  // Game over
        board->revealAllMines();  // Show where the mines were
    }

    
    void GameplayManager::render(sf::RenderWindow& window)
    {
        window.draw(background_sprite);
        board->render(window);
    }

    void GameplayManager::initialize()
    {
        initializeBackground();
        initializeVariables();
    }

    void GameplayManager::updateRemainingTime() {
        remaining_time -= Time::TimeManager::getDeltaTime();  // Decrease time
        processTimeOver();  // Check if time's up
    }

    void GameplayManager::processTimeOver() {
        if (remaining_time <= 0) {
            remaining_time = 0; // Don't go negative
            game_result = GameResult::LOST; // Game over!
        }
    }
    void GameplayManager::initializeBackground()
    {
        if (!background_texture.loadFromFile(background_texture_path)) {
            std::cerr << "Failed to load background texture!\n";
        }
        background_sprite.setTexture(background_texture);
        background_sprite.setColor(sf::Color(255, 255, 255, background_alpha));
    }

    void GameplayManager::initializeVariables()
    {
        board = new Board(this);
        remaining_time = max_level_duration;  // Start with full time
    }

    bool GameplayManager::hasGameEnded() {
        return game_result != GameResult::NONE;
    }
    
    void GameplayManager::setGameResult(GameResult gameResult) 
    { 
        this->game_result = gameResult; 
    }

 
}