#pragma once
#include <array>

template <size_t size, size_t prevSize>
struct Diff
{
	std::array<double, size> biases;
	std::array<std::array<double, prevSize>, size> weights;

	Diff()
	{
		biases.fill(0.0);
		for (auto& i : weights)
		{
			i.fill(0.0);
		}
	}
};

template <class T>
struct Patch
{
	Diff<T::cSize, T::pSize> first;

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
};

template <class T>
	requires requires {
		T().rest;
	}
struct Patch<T>
{
	Diff<T::cSize, T::pSize> first;
	Patch<decltype(T().rest)> rest;

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
};