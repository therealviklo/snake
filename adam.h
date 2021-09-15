#pragma once
#include <cstddef>
#include <array>

namespace AdamParams
{
	constexpr inline double alpha = 0.0001;
	constexpr inline double beta1 = 0.9;
	constexpr inline double beta2 = 0.999;
	constexpr inline double epsilon = 1e-7;
}

template <size_t prevSize, size_t size>
struct MomentVecPair
{
	std::array<double, size> bm;
	std::array<std::array<double, prevSize>, size> wm;
	std::array<double, size> bv;
	std::array<std::array<double, prevSize>, size> wv;

	MomentVecPair()
	{
		bm.fill(0.0);
		for (auto& i : wm)
		{
			i.fill(0.0);
		}
		bv.fill(0.0);
		for (auto& i : wv)
		{
			i.fill(0.0);
		}
	}
};

template <size_t prevSize, size_t f, size_t ...r>
struct MomentVecs
{
	MomentVecPair<prevSize, f> first;
	MomentVecs<f, r...> rest;
};

template <size_t prevSize, size_t f>
struct MomentVecs<prevSize, f>
{	
	MomentVecPair<prevSize, f> first;
};

template <size_t prevSize, size_t f, size_t ...r>
struct Adam
{
	size_t t = 1;
	MomentVecs<prevSize, f, r...> mv;
};

inline void updatem(double& m, double d)
{
	m = std::clamp(AdamParams::beta1 * m + (1.0 - AdamParams::beta1) * d, -DBL_MAX, DBL_MAX);
	if (std::isnan(m))
	{
		std::printf("mnan %g\n", d);
		std::exit(0);
	}
}

inline void updatev(double& v, double d)
{
	v = std::clamp(AdamParams::beta2 * v + (1.0 - AdamParams::beta2) * d * d, -DBL_MAX, DBL_MAX);
	if (std::isnan(v))
	{
		std::printf("vnan %g\n", d);
		std::exit(0);
	}
}

inline double calcalphat(size_t t)
{
	return AdamParams::alpha * std::sqrt(1.0 - std::pow(AdamParams::beta2, t)) / (1.0 - std::pow(AdamParams::beta1, t));
}

inline void updateparam(double& param, double alphat, double m, double v)
{
	if (std::isnan(param))
	{
		std::puts("paramnan1");
		std::exit(0);
	}
	param = std::clamp(param - alphat * m / (std::sqrt(v) + AdamParams::epsilon), -DBL_MAX, DBL_MAX);
	if (std::isnan(param))
	{
		std::printf("paramnan2 %g %g %g\n", alphat, m, v);
		std::exit(0);
	}
}