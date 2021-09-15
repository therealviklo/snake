#include "saver.h"

void Saver::threadLoop()
{
	std::unique_lock<std::mutex> ul(runningm);
	while (!cv.wait_for(ul, std::chrono::seconds(30), [&]() -> bool { return !running; }))
	{
		std::puts("Sparar ...");
		{
			std::lock_guard const lg(netm);
			net.store("actor.dat");
		}
		std::puts("Sparat.");
	}
}

Saver::Saver(const Actor& net, std::mutex& netm) :
	running(true),
	net(net),
	netm(netm),
	t(&Saver::threadLoop, this) {}

Saver::~Saver()
{
	{
		std::lock_guard const lg(runningm);
		running = false;
		cv.notify_all();
	}
	t.join();
}

void PPOSaver::threadLoop()
{
	std::unique_lock<std::mutex> ul(runningm);
	while (!cv.wait_for(ul, std::chrono::seconds(30), [&]() -> bool { return !running; }))
	{
		std::puts("Sparar ...");
		{
			std::lock_guard const lg(netm);
			net.store("actor.dat");
			critic.store("critic.dat");
		}
		std::puts("Sparat.");
	}
}

PPOSaver::PPOSaver(const Actor::Conns& net, const Critic& critic, std::mutex& netm) :
	running(true),
	net(net),
	critic(critic),
	netm(netm),
	t(&PPOSaver::threadLoop, this) {}

PPOSaver::~PPOSaver()
{
	{
		std::lock_guard const lg(runningm);
		running = false;
		cv.notify_all();
	}
	t.join();
}

void QSaver::threadLoop()
{
	std::unique_lock<std::mutex> ul(runningm);
	while (!cv.wait_for(ul, std::chrono::seconds(30), [&]() -> bool { return !running; }))
	{
		std::puts("Sparar ...");
		{
			std::lock_guard const lg(netm);
			net.store("eval.dat");
		}
		std::puts("Sparat.");
	}
}

QSaver::QSaver(const Evaluator::Conns& net, std::mutex& netm) :
	running(true),
	net(net),
	netm(netm),
	t(&QSaver::threadLoop, this) {}

QSaver::~QSaver()
{
	{
		std::lock_guard const lg(runningm);
		running = false;
		cv.notify_all();
	}
	t.join();
}