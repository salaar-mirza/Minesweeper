#pragma once
#include <SFML/Graphics/Sprite.hpp>
#include <SFML/Graphics/Texture.hpp>
#include <SFML/System/Vector2.hpp>
#include <SFML/Graphics/Rect.hpp>
#include <functional>
#include <string>

namespace Event
{
    class EventPollingManager;
}
namespace sf
{
    class RenderWindow;
}

namespace UIElements {

    enum class MouseButtonType
    {
        LEFT_MOUSE_BUTTON,
        RIGHT_MOUSE_BUTTON
    };
    
    // A reusable UI button class that can be configured with a texture and a callback function.
    // It handles mouse interactions and invokes the callback when clicked.
    class Button {
    public:
        Button(const std::string& texture_path, const sf::Vector2f& position, float width, float height);

        // Configuration
        void setTextureRect(const sf::IntRect& rect);
        using CallbackFunction = std::function<void(MouseButtonType)>;
        void registerCallbackFunction(CallbackFunction button_callback);
        
        // Core Logic & Rendering
        void handleButtonInteractions(const Event::EventPollingManager& event_manager, const sf::RenderWindow& window);
        void render(sf::RenderWindow& window) const;

    private:
        sf::Texture button_texture;
        sf::Sprite buttonSprite;

        CallbackFunction callback_function = nullptr;

        // Internal helpers
        void initialize(const std::string& texture_path, const sf::Vector2f& position, float width, float height);
        bool isMouseOnSprite(const Event::EventPollingManager& event_manager, const sf::RenderWindow& window) const;
    };
}