#pragma once
#include <array>
#include <cmath>
#include <cfloat>
#include "rand.h"
#include "patch.h"
#include "activation.h"

template <size_t size, size_t prevSize>
struct Connections
{
	std::array<double, size> biases;
	std::array<std::array<double, prevSize>, size> weights;
};

template <class Act, class LastAct, size_t prevSize, size_t f, size_t ...r>
struct ConnectionLayers
{
	inline static constexpr size_t cSize = f;
	inline static constexpr size_t pSize = prevSize;

	Connections<f, prevSize> first;
	ConnectionLayers<Act, LastAct, f, r...> rest;

	template <size_t n>
	constexpr auto& l() noexcept
	{
		if constexpr (n != 0)
		{
			return rest.template l<n - 1>();
		}
		else
		{
			return first;
		}
	}

	template <size_t n>
	constexpr const auto& l() const noexcept
	{
		if constexpr (n != 0)
		{
			return rest.template l<n - 1>();
		}
		else
		{
			return first;
		}
	}

	void randomise()
	{
		for (auto& i : first.biases)
		{
			i = initfun<Act>(prevSize);
		}
		for (auto& i : first.weights)
		{
			for (auto& j : i)
			{
				j = initfun<Act>(prevSize);
			}
		}

		rest.randomise();
	}

	template <class PatchT>
	void applyPatch(const Patch<PatchT>& patch)
	{
		for (size_t i = 0; i < first.biases.size(); ++i)
		{
			const auto prevfirstbiasesi = first.biases[i];
			first.biases[i] = std::clamp(first.biases[i] + patch.first.biases[i], -DBL_MAX, DBL_MAX);
			if (std::isnan(first.biases[i]))
			{
				std::printf("NAN4, %g %g\n", prevfirstbiasesi, patch.first.biases[i]);
				std::exit(0);
			}
		}

		for (size_t i = 0; i < first.weights.size(); ++i)
		{
			for (size_t j = 0; j < first.weights[i].size(); ++j)
			{
				first.weights[i][j] = std::clamp(first.weights[i][j] + patch.first.weights[i][j], -DBL_MAX, DBL_MAX);
				if (std::isnan(first.weights[i][j]))
				{
					std::puts("NAN5");
					std::exit(0);
				}
			}
		}

		rest.applyPatch(patch.rest);
	}

	void printWeights(size_t n) const noexcept
	{
		std::printf("Lager %zu:\n", ++n);
		for (const auto& i : first.weights)
		{
			for (const auto& j : i)
			{
				std::printf("%g\n", j);
			}
		}
		rest.printWeights(n);
	}

	void printBiases(size_t n) const noexcept
	{
		std::printf("Lager %zu:\n", ++n);
		for (const auto& i : first.biases)
		{
			std::printf("%g\n", i);
		}
		rest.printBiases(n);
	}

	void load(const char* fname)
	{
		if (std::filesystem::exists(fname))
			readFromFile(*this, fname);
		else
			randomise();
	}
	void store(const char* fname) const
	{
		writeToFile(*this, fname);
	}
};

template <class Act, class LastAct, size_t prevSize, size_t f>
struct ConnectionLayers<Act, LastAct, prevSize, f>
{
	inline static constexpr size_t cSize = f;
	inline static constexpr size_t pSize = prevSize;

	Connections<f, prevSize> first;

	template <size_t n>
	constexpr auto& l() noexcept
	{
		return first;
	}

	template <size_t n>
	constexpr const auto& l() const noexcept
	{
		return first;
	}

	void randomise()
	{
		for (auto& i : first.biases)
		{
			i = initfun<LastAct>(prevSize);
		}
		for (auto& i : first.weights)
		{
			for (auto& j : i)
			{
				j = initfun<LastAct>(prevSize);
			}
		}
	}

	template <class PatchT>
	void applyPatch(const Patch<PatchT>& patch)
	{
		for (size_t i = 0; i < first.biases.size(); ++i)
		{
			first.biases[i] = std::clamp(first.biases[i] + patch.first.biases[i], -DBL_MAX, DBL_MAX);
			if (std::isnan(first.biases[i]))
			{
				std::printf("NAN2 %g\n", patch.first.biases[i]);
				std::exit(0);
			}
		}

		for (size_t i = 0; i < first.weights.size(); ++i)
		{
			for (size_t j = 0; j < first.weights[i].size(); ++j)
			{
				first.weights[i][j] = std::clamp(first.weights[i][j] + patch.first.weights[i][j], -DBL_MAX, DBL_MAX);
				if (std::isnan(first.weights[i][j]))
				{
					std::puts("NAN3");
					std::exit(0);
				}
			}
		}
	}

	void printWeights(size_t n) const noexcept
	{
		std::printf("Lager %zu:\n", ++n);
		for (const auto& i : first.weights)
		{
			for (const auto& j : i)
			{
				std::printf("%g\n", j);
			}
		}
	}

	void printBiases(size_t n) const noexcept
	{
		std::printf("Lager %zu:\n", ++n);
		for (const auto& i : first.biases)
		{
			std::printf("%g\n", i);
		}
	}

	void load(const char* fname)
	{
		if (std::filesystem::exists(fname))
			readFromFile(*this, fname);
		else
			randomise();
	}
	void store(const char* fname) const
	{
		writeToFile(*this, fname);
	}
};