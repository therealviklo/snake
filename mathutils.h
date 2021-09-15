#pragma once
#include <cmath>
#include <vector>
#include <cfloat>

inline void normalise(std::vector<double>& nums)
{
	double mean = 0.0;
	for (const auto& i : nums)
	{
		mean += i;
	}
	mean /= nums.size();

	double stddev = 0.0;
	for (auto& i : nums)
	{
		stddev += [](double x) -> double { return x * x; }(i = i - mean);
	}
	stddev /= nums.size();
	stddev = std::sqrt(stddev);

	for (auto& i : nums)
	{
		i /= (stddev + 1e-10);
	}
}

inline double square(double x)
{
	return x * x;
}