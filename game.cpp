#include "game.h"

void Game::moveApple()
{
	auto appleIsInSnake = [&]() -> bool {
		for (const auto& i : snake)
		{
			if (i.x == apple.x && i.y == apple.y)
				return true;
		}
		return false;
	};
	do
	{
		apple.x = randz() % gsize;
		apple.y = randz() % gsize;
	} while (appleIsInSnake());
}

Game::Game() :
	snake{[]() -> std::vector<Pz> {
		const auto pos = Pz{randz() % gsize, randz() % gsize};
		return {{pos, pos, pos}};
	}()},
	dir(Dir::east),
	timeout(0),
	score(0)
{
	moveApple();
}

Outcome Game::logic(Dir dir)
{
	if (++timeout == 200 || score == gsize * gsize) return Outcome::timeout;

	this->dir = dir;
	
	const auto tailpos = *snake.rbegin();
	for (auto i = snake.rbegin(), next = i + 1; next != snake.rend(); i = next, ++next)
	{
		*i = *next;
	}

	switch (this->dir)
	{
		case Dir::east : snake[0].x++; break;
		case Dir::north: snake[0].y--; break;
		case Dir::west : snake[0].x--; break;
		case Dir::south: snake[0].y++; break;
	}

	auto hasBittenTail = [&]() -> bool {
		for (auto i = snake.begin() + 1; i != snake.end(); ++i)
		{
			if (i->x == snake[0].x && i->y == snake[0].y)
				return true;
		}
		return false;
	};

	if (snake[0].x >= gsize || snake[0].y >= gsize || hasBittenTail())
		return Outcome::death;
	else if (snake[0].x == apple.x && snake[0].y == apple.y)
		return timeout = 0, snake.push_back(tailpos), moveApple(), ++score, Outcome::eat;
	return Outcome::nothing;
}

void Game::fillInputLayer(double* layer)
{
	auto pointInSnake = [&](size_t x, size_t y) -> bool {
		for (const auto& i : snake)
		{
			if (x == i.x && y == i.y)
				return true;
		}
		return false;
	};
	for (size_t y = 0; y < vsize; ++y)
	{
		for (size_t x = 0; x < vsize; ++x)
		{
			const size_t realx = snake[0].x - vsize / 2 + x;
			const size_t realy = snake[0].y - vsize / 2 + y;
			if (realx >= gsize || realy >= gsize)
				layer[y * vsize + x] = 1.0;
			else
				layer[y * vsize + x] = 0.0;
		}
	}
	for (size_t y = 0; y < vsize; ++y)
	{
		for (size_t x = 0; x < vsize; ++x)
		{
			const size_t realx = snake[0].x - vsize / 2 + x;
			const size_t realy = snake[0].y - vsize / 2 + y;
			if (apple.x == realx || apple.y == realy)
				layer[vsize * vsize + y * vsize + x] = 1.0;
			else
				layer[vsize * vsize + y * vsize + x] = 0.0;
		}
	}
	for (size_t y = 0; y < vsize; ++y)
	{
		for (size_t x = 0; x < vsize; ++x)
		{
			const size_t realx = snake[0].x - vsize / 2 + x;
			const size_t realy = snake[0].y - vsize / 2 + y;
			if (pointInSnake(realx, realy))
				layer[vsize * vsize * 2 + y * vsize + x] = 1.0;
			else
				layer[vsize * vsize * 2 + y * vsize + x] = 0.0;
		}
	}
	layer[vsize * vsize * 3 + 0] = std::max(0.0, snake[0].x / (double)gsize - apple.x / (double)gsize);
	layer[vsize * vsize * 3 + 1] = std::max(0.0, apple.x / (double)gsize - snake[0].x / (double)gsize);
	layer[vsize * vsize * 3 + 2] = std::max(0.0, snake[0].y / (double)gsize - apple.y / (double)gsize);
	layer[vsize * vsize * 3 + 3] = std::max(0.0, apple.y / (double)gsize - snake[0].y / (double)gsize);
}

void Game::print()
{
	auto pointInSnake = [&](size_t x, size_t y) -> bool {
		for (const auto& i : snake)
		{
			if (x == i.x && y == i.y)
				return true;
		}
		return false;
	};
	std::string pstring = "\e[1;1H";
	for (size_t y = 0; y < gsize; ++y)
	{
		for (size_t x = 0; x < gsize; ++x)
		{
			if (pointInSnake(x, y))
				pstring += '#';
			else if (x == apple.x && y == apple.y)
				pstring += 'O';
			else
				pstring += ' ';
		}
		pstring += "|\n";
	}
	for (size_t i = 0; i < gsize; ++i)
		pstring += '-';
	pstring += /*"/\e[0J"*/'/';
	std::puts(pstring.c_str());
}