#include "rollout.h"

const unsigned thrcnt = std::max<unsigned>(1u, std::thread::hardware_concurrency() - 1);
const unsigned maxtispthread = 1u;

void Rollout::threadLoop(unsigned id)
{
	Actor actor;
	std::vector<TimestepInfo> ti;
	std::unique_lock ul(m);
	while (cv.wait(ul, [&]() -> bool { return tis[id].empty() || !running; }), running)
	{
		ul.unlock();
		{
			const std::lock_guard lg(netm);
			actor.loadConns(actorconns);
		}
		for (unsigned i = 0; i < maxtispthread; ++i)
		{
			Game game;
			long long dist = game.getDistance();
			while (true)
			{
				ti.emplace_back();
				game.fillInputLayer(actor.inputs.data());
				actor.run();
				ti.rbegin()->inputs = actor.inputs;
				const size_t act = actor.getProbOutput(0.1);
				ti.rbegin()->actionTaken = act;
				ti.rbegin()->prob = actor.getRealProb(act);
				switch (game.logic((Dir)act))
				{
					case Outcome::death:
					{
						ti.rbegin()->reward = -10.0;
					}
					goto gameOver;
					case Outcome::timeout:
					{
						ti.rbegin()->reward = 0.0;
					}
					goto gameOver;
					case Outcome::eat:
					{
						dist = game.getDistance();
						ti.rbegin()->reward = 1.0;
					}
					break;
					case Outcome::nothing:
					{
						const auto newDist = game.getDistance();
						if (newDist < dist)
						{
							dist = newDist;
							ti.rbegin()->reward = 1.0;
						}
						else if (newDist > dist)
						{
							dist = newDist;
							ti.rbegin()->reward = -1.0;
						}
						else
						{
							ti.rbegin()->reward = 0.0;
						}
					}
					break;
					default:
					{
						ti.rbegin()->reward = 0.0;
					}
					break;
				}
			}
		gameOver:
			double currRew = 0.0;
			for (auto i = ti.rbegin(); i != ti.rend(); ++i)
			{
				i->reward = (currRew = currRew * gamma + i->reward);
			}
			const std::lock_guard lg(m);
			tis[id].emplace_back();
			tis[id].rbegin()->swap(ti);
		}
		ul.lock();
		cv.notify_all();
	}
}

Rollout::Rollout(std::mutex& netm, Actor::Conns& actorconns, double gamma) :
	gamma(gamma),
	running(true),
	netm(netm),
	actorconns(actorconns),
	tis(thrcnt)
{
	for (auto& i : tis)
	{
		i.reserve(maxtispthread);
	}
	for (unsigned i = 0; i < thrcnt; ++i)
	{
		thrs.emplace_back(&Rollout::threadLoop, this, i);
	}
}

Rollout::~Rollout()
{
	{
		std::lock_guard const lg(m);
		running = false;
		cv.notify_all();
	}
	for (auto& i : thrs)
	{
		i.join();
	}
}

std::vector<std::vector<std::vector<TimestepInfo>>> Rollout::get()
{
	std::vector<std::vector<std::vector<TimestepInfo>>> op(thrcnt);
	for (auto& i : op)
	{
		i.reserve(maxtispthread);
	}
	{
		std::unique_lock ul(m);
		cv.wait(ul, [&]() -> bool {
			for (const auto& i : tis)
			{
				if (i.size() != maxtispthread)
					return false;
			}
			return true;
		});
		op.swap(tis);
		cv.notify_all();
	}
	return op;
}

void QRollout::threadLoop(unsigned id)
{
	Evaluator eval;
	std::vector<QTimestepInfo> ti;
	std::unique_lock ul(m);
	while (cv.wait(ul, [&]() -> bool { return tis[id].empty() || !running; }), running)
	{
		ul.unlock();
		{
			const std::lock_guard lg(netm);
			eval.loadConns(evalconns);
		}
		for (unsigned i = 0; i < maxtispthread; ++i)
		{
			Game game;
			long long dist = game.getDistance();
			while (true)
			{
				ti.emplace_back();
				game.fillInputLayer(eval.inputs.data());
				eval.run();
				ti.rbegin()->inputs = eval.inputs;
				const size_t act = eval.getProbOutput(0.1);
				ti.rbegin()->actionTaken = act;
				switch (game.logic((Dir)act))
				{
					case Outcome::death:
					{
						ti.rbegin()->reward = -10.0;
					}
					goto gameOver;
					case Outcome::timeout:
					{
						ti.rbegin()->reward = 0.0;
					}
					goto gameOver;
					case Outcome::eat:
					{
						dist = game.getDistance();
						ti.rbegin()->reward = 1.0;
					}
					break;
					case Outcome::nothing:
					{
						const auto newDist = game.getDistance();
						if (newDist < dist)
						{
							dist = newDist;
							ti.rbegin()->reward = 1.0;
						}
						else if (newDist > dist)
						{
							dist = newDist;
							ti.rbegin()->reward = -1.0;
						}
						else
						{
							ti.rbegin()->reward = 0.0;
						}
					}
					break;
					default:
					{
						ti.rbegin()->reward = 0.0;
					}
					break;
				}
			}
		gameOver:
			double currRew = 0.0;
			for (auto i = ti.rbegin(); i != ti.rend(); ++i)
			{
				i->reward = (currRew = currRew * gamma + i->reward);
			}
			const std::lock_guard lg(m);
			tis[id].emplace_back();
			tis[id].rbegin()->swap(ti);
		}
		ul.lock();
		cv.notify_all();
	}
}

QRollout::QRollout(std::mutex& netm, Evaluator::Conns& evalconns, double gamma) :
	gamma(gamma),
	running(true),
	netm(netm),
	evalconns(evalconns),
	tis(thrcnt)
{
	for (auto& i : tis)
	{
		i.reserve(maxtispthread);
	}
	for (unsigned i = 0; i < thrcnt; ++i)
	{
		thrs.emplace_back(&QRollout::threadLoop, this, i);
	}
}

QRollout::~QRollout()
{
	{
		std::lock_guard const lg(m);
		running = false;
		cv.notify_all();
	}
	for (auto& i : thrs)
	{
		i.join();
	}
}

std::vector<std::vector<std::vector<QTimestepInfo>>> QRollout::get()
{
	std::vector<std::vector<std::vector<QTimestepInfo>>> op(thrcnt);
	for (auto& i : op)
	{
		i.reserve(maxtispthread);
	}
	{
		std::unique_lock ul(m);
		cv.wait(ul, [&]() -> bool {
			for (const auto& i : tis)
			{
				if (i.size() != maxtispthread)
					return false;
			}
			return true;
		});
		op.swap(tis);
		cv.notify_all();
	}
	return op;
}