#pragma once
#include "../../header/UI/UIElements/Button/Button.h"
#include <SFML/Graphics/Texture.hpp>
#include <SFML/Graphics/Sprite.hpp>
#include <SFML/System/Vector2.hpp>

namespace sf { class RenderWindow; }

namespace Event
{
    class EventPollingManager;
}

namespace UI {
    class MainMenuManager {
    public:
        MainMenuManager(sf::RenderWindow* window);
        ~MainMenuManager();

        void update(Event::EventPollingManager& eventManager);
        void render();

    private:
        // Constants
        static constexpr const char* background_texture_path = "assets/textures/minesweeper_bg.png";
        static constexpr const char* play_button_texture_path = "assets/textures/play_button.png";
        static constexpr const char* quit_button_texture_path = "assets/textures/quit_button.png";

        static constexpr float button_width = 300.f;
        static constexpr float button_height = 100.f;
        static constexpr float play_button_y_position = 600.f;
        static constexpr float quit_button_y_position = 750.f;
        static constexpr float background_alpha = 85.f;

        // Objects
        sf::RenderWindow* game_window;
        sf::Texture background_texture;
        sf::Sprite background_sprite;

        UIElements::Button* play_button;
        UIElements::Button* quit_button;

        // Initialization
        void initialize();
        void initializeBackground();
        void initializeButtons();
        void registerButtonCallbacks();

        // Helpers
        void show();
        void checkForButtonClicks(Event::EventPollingManager& eventManager);
        void playButtonCallback(UIElements::MouseButtonType mouse_button_type);
        void quitButtonCallback(UIElements::MouseButtonType mouse_button_type);
        sf::Vector2f getButtonPosition(float offsetX, float offsetY) const;
    };
}
