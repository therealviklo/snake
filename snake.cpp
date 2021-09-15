#include <cstring>
#include <thread>
#include <chrono>
#include <vector>
#include <memory>
#include "net.h"
#include "game.h"
#include "saver.h"
#include "ppo.h"
#include "q.h"

using namespace std::literals;

int main(int argc, char* argv[])
{
	if (argc < 2)
	{
	tooFewArguments:
		std::puts("För få argument");
		return 0;
	}

	if (std::strcmp(argv[1], "run") == 0)
	{
	restart:
		bool fastmode = false;
		bool probmode = false;
		bool qmode = false;

		for (int i = 2; i < argc; i++)
		{
			if (argv[i][0] == '-')
			{
				switch (argv[i][1])
				{
					case 'f':
					{
						fastmode = true;
					}
					continue;
					case 'p':
					{
						probmode = true;
					}
					continue;
					case 'q':
					{
						qmode = true;
					}
					continue;
				}
			}
			std::printf("Okänt argument: %s\n", argv[i]);
			return 0;
		}

		if (qmode)
		{
			Game game;
			std::unique_ptr<Evaluator> net = std::make_unique<Evaluator>();
			net->load("actor.dat");

			std::printf("\e[2J");
			auto fstart = std::chrono::steady_clock::now();
			while (true)
			{
				game.fillInputLayer(net->inputs.data());
				net->run();
				const auto op = probmode ? net->getProbOutput(0.1) : net->getTopOutput();
				if (const auto outcome = game.logic((Dir)op); outcome == Outcome::death || outcome == Outcome::timeout)
					goto restart;
				game.print();

				for (size_t i = 0; i < 4; ++i)
				{
					if (i == op)
						std::printf("%g←     \n", net->getProb(i));
					else
						std::printf("%g      \n", net->getProb(i));
				}

				std::fflush(stdout);
				std::this_thread::sleep_until(
					fstart += std::chrono::duration_cast<std::chrono::steady_clock::duration>(fastmode ? 0.1s : 0.5s)
				);
			}
		}
		else
		{
			Game game;
			std::unique_ptr<Actor> net = std::make_unique<Actor>();
			net->load("actor.dat");
			std::unique_ptr<Critic> critic = std::make_unique<Critic>();
			critic->load("critic.dat");

			std::printf("\e[2J");
			auto fstart = std::chrono::steady_clock::now();
			while (true)
			{
				game.fillInputLayer(net->inputs.data());
				net->run();
				game.print();
				const auto op = probmode ? net->getProbOutput(0.1) : net->getTopOutput();

				for (size_t i = 0; i < 4; ++i)
				{
					if (i == op)
						std::printf("%g←     \n", net->getProb(i));
					else
						std::printf("%g      \n", net->getProb(i));
				}

				game.fillInputLayer(critic->inputs.data());
				critic->run();
				std::printf("V: %g\n", critic->getProb(0));
				
				std::fflush(stdout);
				
				std::this_thread::sleep_until(
					fstart += std::chrono::duration_cast<std::chrono::steady_clock::duration>(fastmode ? 0.1s : 0.5s)
				);

				if (const auto outcome = game.logic((Dir)op); outcome == Outcome::death || outcome == Outcome::timeout)
					goto restart;

			}
		}
	}
	else if (std::strcmp(argv[1], "train") == 0)
	{
		if (argc < 3) goto tooFewArguments;

		if (std::strcmp(argv[2], "q") == 0)
		{
			std::unique_ptr<Q> q = std::make_unique<Q>();
			while (true) q->learn();
		}
		else if (std::strcmp(argv[2], "ppo") == 0)
		{
			std::unique_ptr<PPO> ppo = std::make_unique<PPO>();
			while (true) ppo->learn();
		}
		else
		{
			std::printf("Okänt argument \"%s\"\n", argv[2]);
		}
	}
	else if (std::strcmp(argv[1], "print") == 0)
	{
		if (argc < 3) goto tooFewArguments;

		if (std::strcmp(argv[2], "net") == 0)
		{
			Game game;
			std::unique_ptr<Actor> net = std::make_unique<Actor>();
			net->load("actor.dat");
			game.fillInputLayer(net->inputs.data());
			net->run();
			net->print();
		}
		else if (std::strcmp(argv[2], "critic") == 0)
		{
			Game game;
			std::unique_ptr<Critic> critic = std::make_unique<Critic>();
			critic->load("critic.dat");
			game.fillInputLayer(critic->inputs.data());
			critic->run();
			critic->print();
		}
		else
		{
			std::printf("Okänt argument \"%s\"\n", argv[2]);
		}
	}
	else if (std::strcmp(argv[1], "clear") == 0)
	{
		std::remove("actor.dat");
		std::remove("eval.dat");
		std::remove("critic.dat");
	}
	else
	{
		std::printf("Okänt argument \"%s\"\n", argv[1]);
	}
}