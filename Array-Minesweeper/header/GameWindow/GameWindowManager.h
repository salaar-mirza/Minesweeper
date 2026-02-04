#pragma once
#include <SFML/Graphics.hpp>

namespace GameWindow
{
	class GameWindowManager
	{
	private:
		// We avoid using 'const' for member variables. Using 'const' on data members
		// can make the class non-assignable (by deleting the copy-assignment operator),
		// which can cause issues with object management and flexibility. This also allows
		// for future features like a settings menu to change these values at runtime.

		int game_window_width = 1920;
		int game_window_height = 1080;
		int frame_rate = 60;
		std::string game_window_title = "Outscal Presents - Minesweeper";
		sf::Color window_color = sf::Color(200, 200, 0, 255);

		sf::RenderWindow* game_window;
		sf::VideoMode video_mode;

		void initialize();
		sf::RenderWindow* createGameWindow();
		void setFrameRate(int);
		void configureVideoMode();
		void onDestroy();

	public:
		GameWindowManager();
		~GameWindowManager();

		bool isGameWindowOpen();
		sf::RenderWindow* getGameWindow();

		void update();
		void render();
	};
}