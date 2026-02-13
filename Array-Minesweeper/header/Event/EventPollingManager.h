#pragma once
#include <SFML/Window/Event.hpp>

namespace sf
{
    class RenderWindow;
}

namespace Event
{
    // Manages the state of a mouse button over multiple frames.
    enum class MouseButtonState
    {
        PRESSED,
        HELD,
        RELEASED,
    };

    class EventPollingManager
    // This class is responsible for polling all SFML events, processing them,
    // and providing a clean interface for other parts of the game to query input states
    // (e.g., "was the left mouse button pressed this frame?").
    {
    public:
        EventPollingManager(sf::RenderWindow* window);
        ~EventPollingManager();

        // Core update functions
        void processEvents();
        void update();

        // State query functions
        bool pressedEscapeKey() const;
        bool pressedLeftMouseButton() const;
        bool pressedRightMouseButton() const;
        sf::Vector2i getMousePosition() const;

    private:
        sf::Event game_event;
        sf::RenderWindow* game_window;

        MouseButtonState left_mouse_button_state;
        MouseButtonState right_mouse_button_state;

        void initializeVariables(sf::RenderWindow* window);
        void updateMouseButtonState(MouseButtonState& button_state, sf::Mouse::Button button_type);
        
        // Event processing helpers
        bool hasQuitGame() const;
        bool isKeyboardEvent() const;
        bool isGameWindowOpen() const;
        bool gameWindowWasClosed() const;
    };
}