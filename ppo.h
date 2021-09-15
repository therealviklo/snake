#pragma once
#include "net.h"
#include "game.h"
#include "saver.h"
#include "rollout.h"

class PPO
{
private:
	std::mutex m;
	Actor::Conns actorconns;
	Critic::Adam criticadam;
	Critic critic;
	Actor::Adam actoradam;
	Rollout ro;
	PPOSaver s;
public:
	PPO();

	void learn();
};