#include "q.h"

namespace QParams
{
	constexpr inline double gamma = 0.0;
	constexpr inline unsigned epochs = 5;
	constexpr inline double wd = 0.01;
}

Q::Q() :
	s(evalconns, m),
	ro(m, evalconns, QParams::gamma)
{
	evalconns.load("eval.dat");
}

void Q::learn()
{
	static Evaluator eval = [&](){
		std::lock_guard const lg(m);
		Evaluator eval;
		eval.loadConns(evalconns);
		return eval;
	}();

	std::vector<std::vector<std::vector<QTimestepInfo>>> ti = ro.get();

	const double numbackprops = QParams::epochs * [&]() -> size_t {
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

	for (unsigned epoch = 0; epoch < QParams::epochs; ++epoch)
	{
		for (const auto& i : ti)
		{
			for (const auto& j : i)
			{
				for (const auto& k : j)
				{
					Evaluator::Patch patch;
					eval.inputs = k.inputs;
					eval.run();
					std::array<double, 4> desiredChange = {{0.0, 0.0, 0.0, 0.0}};
					desiredChange[k.actionTaken] = -2.0 * (k.reward - eval.getProb(k.actionTaken)) / numbackprops;
					eval.adam(adam, desiredChange, patch, QParams::wd);
					eval.applyPatch(patch);
				}
			}
		}
	}

	{
		const std::lock_guard lg(m);
		eval.storeConns(evalconns);
	}
}