#pragma once
#include <SFML/Graphics/Texture.hpp>
#include <SFML/Graphics/Sprite.hpp>
#include <SFML/System/Vector2.hpp>
#include <string>

namespace sf { class RenderWindow; }

namespace UI {
    class SplashScreenManager {
    public:
        SplashScreenManager(sf::RenderWindow* window);
        ~SplashScreenManager();

        void update();
        void render();

    private:
        // Config
        float logo_width = 600.f;
        float logo_height = 134.f;
        float logo_animation_duration = 2.0f;
        std::string logo_texture_path = "assets/textures/outscal_logo.png";

        // State
        float elapsed_time = 0.0f;

        // Objects
        sf::RenderWindow* game_window;
        sf::Texture logo_texture;
        sf::Sprite logo_sprite;

        void initialize();
        sf::Vector2f getLogoPosition() const;
        void drawLogo();
    };
}