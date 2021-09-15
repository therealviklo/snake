#pragma once
#include <array>
#include <vector>
#include <cstdio>
#include <string>
#include "rand.h"

template <class T>
struct Point
{
	T x;
	T y;
};
using Pz = Point<size_t>;

enum struct Dir : unsigned char
{
	east,
	north,
	west,
	south
};

enum struct Outcome : unsigned char
{
	nothing,
	eat,
	death,
	timeout
};

inline constexpr size_t gsize = 10;
inline constexpr size_t vsize = 5;
class Game
{
private:
	std::vector<Pz> snake;
	Pz apple;
	Dir dir;
	unsigned timeout;
	unsigned score;

	void moveApple();
public:
	Game();

	Outcome logic(Dir dir);
	void fillInputLayer(double* layer);
	void print();

	long long getDistance() const noexcept
	{
		return std::abs((long long)snake[0].x - (long long)apple.x) + std::abs((long long)snake[0].y - (long long)apple.y);
	}
};