#pragma once
#include <SFML/Graphics/Font.hpp>
#include <SFML/Graphics/Text.hpp>
#include <SFML/Graphics/Color.hpp>
#include "../../header/UI/UIElements/Button/Button.h"

namespace sf { class RenderWindow; }

namespace Gameplay
{
    class GameplayManager;
}

namespace Event { class EventPollingManager; }

namespace UI
{
    class GameplayUI {
    public:
        GameplayUI(Gameplay::GameplayManager* gameplay_manager);
        ~GameplayUI();

        void update(int remaining_mines, int remaining_time, Event::EventPollingManager& eventManager, sf::RenderWindow& window);
        void render(sf::RenderWindow& window);

    private:
        // Constants
         static constexpr const char* restartButtonTexturePath = "assets/textures/restart_button.png";
         static constexpr int fontSize = 110;
         sf::Color textColor = sf::Color::Red;

         static constexpr float mineTextTopOffset = 65.f;
         static constexpr float mineTextLeftOffset = 660.f;

         static constexpr float timeTextTopOffset = 65.f;
         static constexpr float timeTextLeftOffset = 1090.f;

         static constexpr float restartButtonTopOffset = 100.f;
         static constexpr float restartButtonLeftOffset = 920.f;

         static constexpr float buttonWidth = 80.f;
         static constexpr float buttonHeight = 80.f;

        // Assets & UI Elements
        sf::Font bubbleBobbleFont;
        sf::Font dsDigibFont;
        sf::Text mineText;
        sf::Text timeText;
        UIElements::Button* restartButton = nullptr;

        // Object Pointers
        Gameplay::GameplayManager* gameplay_manager;
        
        // Initialization
        void initialize(Gameplay::GameplayManager* p_gameplay_manager);
        void initializeTexts();
        void initializeButton();
        void loadFonts();
				
        // Callback System
        void registerButtonCallback();
       void RestartButtonCallback(UIElements::MouseButtonType mouse_button_type);
    };
}
