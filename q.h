#pragma once
#include "net.h"
#include "game.h"
#include "saver.h"
#include "rollout.h"

class Q
{
private:
	std::mutex m;
	Evaluator::Conns evalconns;
	Evaluator::Adam adam;
	QSaver s;
	QRollout ro;
public:
	Q();

	void learn();
};