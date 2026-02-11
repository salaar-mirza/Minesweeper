#include <iostream>
#include "../../header/UI/GameplayUI/GamplayUI.h"
#include "../../header/GameLoop/Gameplay/GameplayManager.h"
#include "../../header/Event/EventPollingManager.h"
#include "../../header/Sound/SoundManager.h"


namespace UI {

    GameplayUI::GameplayUI(Gameplay::GameplayManager* gameplay_manager) 
    { 
        initialize(gameplay_manager); 
    }

    void GameplayUI::initialize(Gameplay::GameplayManager* gameplay_manager)
    {
        this->gameplay_manager = gameplay_manager;
        loadFonts();
        initializeTexts();
        initializeButton();  // Initialize Restart Button
        registerButtonCallback();  // Register callback for the button
    }

    void GameplayUI::loadFonts()
    {
        if (!bubbleBobbleFont.loadFromFile("assets/fonts/bubbleBobble.ttf"))
            std::cerr << "Error loading bubbleBobble font!" << std::endl;
    
        if (!dsDigibFont.loadFromFile("assets/fonts/DS_DIGIB.ttf"))
            std::cerr << "Error loading DS_DIGIB font!" << std::endl;
    }

    void GameplayUI::initializeTexts() {
        // Mine Text
        mineText.setFont(dsDigibFont);
        mineText.setCharacterSize(fontSize);
        mineText.setFillColor(textColor);
        mineText.setPosition(mineTextLeftOffset, mineTextTopOffset);
        mineText.setString("000");  // Default display

        // Time Text
        timeText.setFont(dsDigibFont);
        timeText.setCharacterSize(fontSize);
        timeText.setFillColor(textColor);
        timeText.setPosition(timeTextLeftOffset, timeTextTopOffset);
        timeText.setString("000");
    }

    void GameplayUI::initializeButton() {
        restartButton = new UIElements::Button(restartButtonTexturePath,
                                 sf::Vector2f(restartButtonLeftOffset, restartButtonTopOffset),
                                 buttonWidth, buttonHeight);
    }

    void GameplayUI::registerButtonCallback() {
        restartButton->registerCallbackFunction([this](UIElements::MouseButtonType buttonType) {
            RestartButtonCallback(buttonType);
        });
    }

    void GameplayUI::RestartButtonCallback(UIElements::MouseButtonType mouse_button_type) {
        if (mouse_button_type == UIElements::MouseButtonType::LEFT_MOUSE_BUTTON) {
            Sound::SoundManager::PlaySound(Sound::SoundType::BUTTON_CLICK);
            gameplay_manager->restartGame();  // Restart the game
        }
    }

    void GameplayUI::update(int remaining_mines, int remaining_time, Event::EventPollingManager& eventManager, sf::RenderWindow& window) {
        mineText.setString(std::to_string(remaining_mines));
        timeText.setString(std::to_string(remaining_time));
        restartButton->handleButtonInteractions(eventManager, window);


    }

    void GameplayUI::render(sf::RenderWindow& window) {
        window.draw(mineText);
        window.draw(timeText);
        restartButton->render(window);

    }
}
