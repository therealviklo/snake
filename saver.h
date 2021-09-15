#pragma once
#include <chrono>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <cstdio>
#include "net.h"
#include "fio.h"

class Saver
{
private:
	bool running;
	std::mutex runningm;
	std::condition_variable cv;
	const Actor& net;
	std::mutex& netm;

	std::thread t;

	void threadLoop();
public:
	Saver(const Actor& net, std::mutex& netm);
	~Saver();

	Saver(const Saver&) = delete;
	Saver& operator=(const Saver&) = delete;
};

class PPOSaver
{
private:
	bool running;
	std::mutex runningm;
	std::condition_variable cv;
	const Actor::Conns& net;
	const Critic& critic;
	std::mutex& netm;

	std::thread t;

	void threadLoop();
public:
	PPOSaver(const Actor::Conns& net, const Critic& critic, std::mutex& netm);
	~PPOSaver();

	PPOSaver(const PPOSaver&) = delete;
	PPOSaver& operator=(const PPOSaver&) = delete;
};

class QSaver
{
private:
	bool running;
	std::mutex runningm;
	std::condition_variable cv;
	const Evaluator::Conns& net;
	std::mutex& netm;

	std::thread t;

	void threadLoop();
public:
	QSaver(const Evaluator::Conns& net, std::mutex& netm);
	~QSaver();

	QSaver(const QSaver&) = delete;
	QSaver& operator=(const QSaver&) = delete;
};