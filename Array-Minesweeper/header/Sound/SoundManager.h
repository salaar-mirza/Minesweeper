#pragma once
#include <SFML/Audio.hpp>
#include <string>

namespace Sound
{
    enum class SoundType
    {
        BUTTON_CLICK,
        FLAG,
        EXPLOSION,
        GAME_WON
    };

    // A static utility class for loading and playing all game sounds and music.
    // It provides a simple interface to play sound effects by type and manage background music.
    class SoundManager
    {
    public:
        // Initializes all sound buffers and background music. Should be called once at game startup.
        static void Initialize();

        // Plays a specific sound effect based on its type.
        static void PlaySound(SoundType soundType);

        // Starts playing the background music on a loop.
        static void PlayBackgroundMusic();

    private:
        // Sound Data
        static float backgroundMusicVolume;
        static std::string button_click_path;
        static std::string flag_sound_path;
        static std::string explosion_sound_path;
        static std::string game_won_sound_path;
        static std::string background_path;

        // Sound Objects
        static sf::Music backgroundMusic;
        static sf::SoundBuffer bufferButtonClick;
        static sf::SoundBuffer bufferFlagSound;
        static sf::SoundBuffer bufferExplosion;
        static sf::SoundBuffer bufferGameWon;
        static sf::Sound soundEffect;

        // Private helper methods for initialization
        static void LoadBackgroundMusicFromFile();
        static void LoadSoundFromFile();
    };
}