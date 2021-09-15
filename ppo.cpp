#include "ppo.h"

namespace PPOParams
{
	constexpr inline double gamma = 0.0;
	constexpr inline double epsilon = 0.2;
	constexpr inline unsigned epochs = 5;
	constexpr inline double wd = 0.0001;
}

PPO::PPO() :
	ro(m, actorconns, PPOParams::gamma),
	s(actorconns, critic, m)
{
	actorconns.load("actor.dat");
	critic.load("critic.dat");
}

void PPO::learn()
{
	static Actor actor = [&](){
		std::lock_guard const lg(m);
		Actor actor;
		actor.loadConns(actorconns);
		return actor;
	}();

	std::vector<std::vector<std::vector<TimestepInfo>>> ti = ro.get();
	
	std::vector<double> advantages;
	advantages.reserve(ti.size());
	for (const auto& i : ti)
	{
		for (const auto& j : i)
		{
			for (const auto& k : j)
			{
				critic.inputs = k.inputs;
				critic.run();
				advantages.push_back(k.reward - critic.getProb(0));
			}
		}
	}
	normalise(advantages);

	const double numbackprops = PPOParams::epochs * [&]() -> size_t {
		size_t cnt = 0;
		for (const auto& i : ti)
		{
			for (const auto& j : i)
			{
				cnt += j.size();
			}
		}
		return cnt;
	}();

	for (unsigned epoch = 0; epoch < PPOParams::epochs; ++epoch)
	{
		const double* currAdvantage = advantages.data();
		for (const auto& i : ti)
		{
			for (const auto& j : i)
			{
				for (const auto& k : j)
				{
					Actor::Patch patch;
					actor.inputs = k.inputs;
					actor.run();
					std::array<double, 4> desiredChange;
					desiredChange.fill(0.0);
					desiredChange[k.actionTaken]
						= -ppoclipderiv(
							actor.getRealProb(k.actionTaken),
							k.prob,
							actor.getTotProb(),
							*currAdvantage++,
							PPOParams::epsilon
						) / numbackprops;
					if (std::isinf(desiredChange[k.actionTaken]))
					{
						std::printf("dcinf1 %g %g %g %g %g\n", actor.getRealProb(k.actionTaken), k.prob, actor.getTotProb(), *(currAdvantage - 1), numbackprops);
						std::exit(0);
					}
					actor.adam(actoradam, desiredChange, patch, PPOParams::wd);
					actor.applyPatch(patch);
				}
			}
		}

		for (const auto& i : ti)
		{
			for (const auto& j : i)
			{
				for (const auto& k : j)
				{
					Critic::Patch cpatch;
					critic.inputs = k.inputs;
					critic.run();
					std::array<double, 1> desiredChange;
					desiredChange[0] = -2.0 * (k.reward - critic.getProb(0)) / numbackprops;
					if (std::isinf(desiredChange[0]))
					{
						std::printf("dcinf2  %g %g %g\n", k.reward, critic.getProb(0), numbackprops);
						std::exit(0);
					}
					critic.adam(criticadam, desiredChange, cpatch, PPOParams::wd);
					critic.applyPatch(cpatch);
				}
			}
		}
	}

	{
		const std::lock_guard lg(m);
		actor.storeConns(actorconns);
	}
}