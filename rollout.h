#pragma once
#include <thread>
#include <mutex>
#include <condition_variable>
#include <vector>
#include "net.h"
#include "game.h"

struct TimestepInfo
{
	decltype(Actor::inputs) inputs;
	size_t actionTaken;
	double prob;
	double reward;
};

class Rollout
{
private:
	const double gamma;
	bool running;
	std::mutex& netm;
	Actor::Conns& actorconns;
	std::mutex m;
	std::condition_variable cv;
	std::vector<std::vector<std::vector<TimestepInfo>>> tis;
	std::vector<std::thread> thrs;

	void threadLoop(unsigned id);
public:
	Rollout(std::mutex& netm, Actor::Conns& actorconns, double gamma);
	~Rollout();

	Rollout(const Rollout&) = delete;
	Rollout& operator=(const Rollout&) = delete;

	std::vector<std::vector<std::vector<TimestepInfo>>> get();
};

struct QTimestepInfo
{
	decltype(Actor::inputs) inputs;
	size_t actionTaken;
	double reward;
};

class QRollout
{
private:
	const double gamma;
	bool running;
	std::mutex& netm;
	Evaluator::Conns& evalconns;
	std::mutex m;
	std::condition_variable cv;
	std::vector<std::vector<std::vector<QTimestepInfo>>> tis;
	std::vector<std::thread> thrs;

	void threadLoop(unsigned id);
public:
	QRollout(std::mutex& netm, Evaluator::Conns& evalconns, double gamma);
	~QRollout();

	QRollout(const QRollout&) = delete;
	QRollout& operator=(const QRollout&) = delete;

	std::vector<std::vector<std::vector<QTimestepInfo>>> get();
};