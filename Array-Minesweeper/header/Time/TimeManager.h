#pragma once
#include <chrono>

namespace Time
{
    // A static utility class for managing frame-independent time.
    // It calculates delta time (the time elapsed since the last frame)
    // to ensure game logic and animations run smoothly regardless of frame rate.
    class TimeManager
    {
    public:
        // Initializes the timer. Should be called once at the start of the game.
        static void initialize();

        // Updates the timer. Should be called once per frame.
        static void update();

        // Returns the time in seconds that has passed since the last frame.
        static float getDeltaTime();

    private:
        static std::chrono::time_point<std::chrono::steady_clock> previous_time;
        static float delta_time;

        static void updateDeltaTime();
        static float calculateDeltaTime();
        static void updatePreviousTime();
    };
}