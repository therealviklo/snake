#pragma once
#include "mathutils.h"
#include "rand.h"

struct Sig
{
	static double act(double x)
	{
		return 1.0 / (1.0 + std::exp(-x));
	}

	static double deriv(double x)
	{
		const double emx = std::exp(-x);
		if (std::isinf(emx)) return DBL_EPSILON;
		return emx / [](double x) -> double { return x * x; }(1.0 + emx);
	}

	static double derivfromact(double x)
	{
		return x - x * x;
	}
};

struct Relu
{
	static double act(double x)
	{
		return x > 0.0 ? x : 0.01 * x;
	}

	static double deriv(double x)
	{
		return x > 0.0 ? 1.0 : 0.01;
	}

	static double derivfromact(double x)
	{
		return x > 0.0 ? 1.0 : 0.01;
	}
};

struct Identity
{
	static double act(double x)
	{
		return x;
	}

	static double deriv(double /*x*/)
	{
		return 1.0;
	}

	static double derivfromact(double /*x*/)
	{
		return 1.0;
	}
};

template <class Act>
double cderiv(double third, double a, double change)
{
	return third * Act::derivfromact(a) * change;
}

template <class Act>
double cderiv(double a, double change)
{
	return Act::derivfromact(a) * change;
}

inline double cpderiv(double third, double change)
{
	return third * change;
}

inline double cpderiv(double change)
{
	return change;
}

inline double prob(double a)
{
	return std::exp(a) + 1e-7;
}

inline double probderiv(double a)
{
	return std::exp(a);
}

inline double g(double epsilon, double a)
{
	return a >= 0 ? (1.0 + epsilon) * a : (1.0 - epsilon) * a;
}

inline double ppoclipderiv(double p, double oldp, double tot, double adv, double epsilon)
{
	const double advoldp = adv / oldp;
	const double lhs = p * advoldp;
	const double rhs = g(epsilon, adv);
	return lhs <= rhs ? std::clamp((adv / tot) * probderiv(p) / oldp, -DBL_MAX, DBL_MAX) : 0.0;
}

template <class Act>
constexpr inline auto initfun = xavier;
template <>
constexpr inline auto initfun<Relu> = hzrs;