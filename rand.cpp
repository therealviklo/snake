#include "rand.h"

std::minstd_rand re(std::random_device{}());

int randi()
{
	return std::uniform_int_distribution<int>{}(re);
}

unsigned randu()
{
	return std::uniform_int_distribution<unsigned>{}(re);
}

size_t randz()
{
	return std::uniform_int_distribution<size_t>{}(re);
}

double rand01()
{
	return randz() / (double)std::numeric_limits<size_t>::max();
}

double rand11()
{
	return std::uniform_real_distribution<double>{-1.0, 1.0}(re);
}

double randd(double min, double max)
{
	return std::uniform_real_distribution<double>{min, max}(re);
}

double randnorm(double mean, double stddev)
{
	return std::normal_distribution<double>{mean, stddev}(re);
}

double xavier(size_t nin)
{
	return std::normal_distribution<double>{0.0, std::sqrt(1.0 / nin)}(re);
}

double hzrs(size_t nin)
{
	return std::normal_distribution<double>{0.0, std::sqrt(2.0 / nin)}(re);
}