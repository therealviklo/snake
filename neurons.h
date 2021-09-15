#pragma once
#include "conns.h"
#include "mathutils.h"
#include "patch.h"
#include "activation.h"
#include "adam.h"

template <size_t size>
using Neurons = std::array<double, size>;

template <class Act, class LastAct, size_t f, size_t ...r>
struct NeuronLayers
{
	Neurons<f> first;
	NeuronLayers<Act, LastAct, r...> rest;

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

	template <size_t prevSize>
	void feedForward(const Neurons<prevSize>& prev, const ConnectionLayers<Act, LastAct, prevSize, f, r...>& conns) noexcept
	{
		first = conns.first.biases;

		for (size_t i = 0; i < first.size(); ++i)
		{
			for (size_t j = 0; j < prev.size(); ++j)
			{
				const auto prevfirsti = first[i];
				first[i] = std::clamp(first[i] + prev[j] * conns.first.weights[i][j], -DBL_MAX, DBL_MAX);
				if (std::isnan(first[i]))
				{
					std::printf("%g, sen %g = %g * %g\n", prevfirsti, first[i], prev[j], conns.first.weights[i][j]);
					std::puts("NAN-1");
					std::exit(0);
				}
			}
			first[i] = Act::act(first[i]);
			if (std::isnan(first[i]))
			{
				std::puts("NAN-2");
				std::exit(0);
			}
		}

		rest.feedForward(first, conns.rest);
	}

	template <size_t prevSize, class PatchT, size_t lastSize>
	void backprop(
		const std::array<double, prevSize>& prev,
		const ConnectionLayers<Act, LastAct, prevSize, f, r...>& conns,
		const std::array<double, lastSize>& inputDiff,
		double wd,
		Patch<PatchT>& out
	) const
	{
		std::array<double, f> desiredChanges;
		rest.backprop(
			first,
			conns.rest,
			inputDiff,
			wd,
			out.rest,
			desiredChanges
		);

		for (size_t i = 0; i < f; ++i)
		{
			out.first.biases[i] = std::clamp(out.first.biases[i] + cderiv<Act>(first[i], desiredChanges[i]), -DBL_MAX, DBL_MAX);
			out.first.biases[i] *= wd;
		}

		for (size_t i = 0; i < f; ++i)
		{
			for (size_t j = 0; j < prevSize; ++j)
			{
				out.first.weights[i][j] = std::clamp(out.first.weights[i][j] + cderiv<Act>(prev[j], first[i], desiredChanges[i]), -DBL_MAX, DBL_MAX);
				out.first.weights[i][j] *= wd;
			}
		}
	}

	template <size_t prevSize, class PatchT, size_t lastSize>
	void backprop(
		const std::array<double, prevSize>& prev,
		const ConnectionLayers<Act, LastAct, prevSize, f, r...>& conns,
		const std::array<double, lastSize>& inputDiff,
		double wd,
		Patch<PatchT>& out,
		std::array<double, prevSize>& nextDesiredChanges
	) const
	{
		std::array<double, f> desiredChanges;
		rest.backprop(
			first,
			conns.rest,
			inputDiff,
			wd,
			out.rest,
			desiredChanges
		);

		for (size_t i = 0; i < f; ++i)
		{
			out.first.biases[i] = std::clamp(out.first.biases[i] + cderiv<Act>(first[i], desiredChanges[i]), -DBL_MAX, DBL_MAX);
			out.first.biases[i] *= wd;
		}

		for (size_t i = 0; i < f; ++i)
		{
			for (size_t j = 0; j < prevSize; ++j)
			{
				out.first.weights[i][j] = std::clamp(out.first.weights[i][j] + cderiv<Act>(prev[j], first[i], desiredChanges[i]), -DBL_MAX, DBL_MAX);
				out.first.weights[i][j] *= wd;
			}
		}

		// Ja, det är j ytterst.
		for (size_t j = 0; j < prevSize; ++j)
		{
			nextDesiredChanges[j] = 0.0;
			for (size_t i = 0; i < f; ++i)
			{
				nextDesiredChanges[j] = std::clamp(nextDesiredChanges[j] + cderiv<Act>(conns.first.weights[i][j], first[i], desiredChanges[i]), -DBL_MAX, DBL_MAX);
			}
		}
	}

	template <size_t prevSize, class PatchT, size_t lastSize>
	void adam(
		MomentVecs<prevSize, f, r...>& mv,
		size_t t,
		const std::array<double, prevSize>& prev,
		const ConnectionLayers<Act, LastAct, prevSize, f, r...>& conns,
		const std::array<double, lastSize>& inputDiff,
		double wd,
		Patch<PatchT>& out,
		std::array<double, prevSize>& nextDesiredChanges
	) const
	{
		std::array<double, f> desiredChanges;
		rest.adam(
			mv.rest,
			t,
			first,
			conns.rest,
			inputDiff,
			wd,
			out.rest,
			desiredChanges
		);

		const double alphat = calcalphat(t);

		for (size_t i = 0; i < f; ++i)
		{
			const double d = cderiv<Act>(first[i], desiredChanges[i]) + wd * conns.first.biases[i];
			if (std::isnan(d))
			{
				std::printf("dnan1 %g %g\n", first[i], desiredChanges[i]);
				std::exit(0);
			}
			updatem(mv.first.bm[i], d);
			updatev(mv.first.bv[i], d);
			updateparam(out.first.biases[i], alphat, mv.first.bm[i], mv.first.bv[i]);
			// out.first.biases[i] *= (1.0 - wd);
		}

		for (size_t i = 0; i < f; ++i)
		{
			for (size_t j = 0; j < prevSize; ++j)
			{
				const double d = cderiv<Act>(prev[j], first[i], desiredChanges[i]) + wd * conns.first.weights[i][j];
				if (std::isnan(d))
				{
					std::printf("dnan2 %g %g %g\n", prev[i], first[i], desiredChanges[i]);
					std::exit(0);
				}
				updatem(mv.first.wm[i][j], d);
				updatev(mv.first.wv[i][j], d);
				updateparam(out.first.weights[i][j], alphat, mv.first.wm[i][j], mv.first.wv[i][j]);
				// out.first.weights[i][j] *= (1.0 - wd);
			}
		}

		// Ja, det är j ytterst.
		for (size_t j = 0; j < prevSize; ++j)
		{
			nextDesiredChanges[j] = 0.0;
			for (size_t i = 0; i < f; ++i)
			{
				nextDesiredChanges[j] = std::clamp(nextDesiredChanges[j] + cderiv<Act>(conns.first.weights[i][j], first[i], desiredChanges[i]), -DBL_MAX, DBL_MAX);
			}
		}
	}

	template <size_t prevSize, class PatchT, size_t lastSize>
	void adam(
		MomentVecs<prevSize, f, r...>& mv,
		size_t t,
		const std::array<double, prevSize>& prev,
		const ConnectionLayers<Act, LastAct, prevSize, f, r...>& conns,
		const std::array<double, lastSize>& inputDiff,
		double wd,
		Patch<PatchT>& out
	) const
	{
		std::array<double, f> desiredChanges;
		rest.adam(
			mv.rest,
			t,
			first,
			conns.rest,
			inputDiff,
			wd,
			out.rest,
			desiredChanges
		);

		const double alphat = calcalphat(t);

		for (size_t i = 0; i < f; ++i)
		{
			const double d = cderiv<Act>(first[i], desiredChanges[i]) + wd * conns.first.biases[i];
			if (std::isnan(d))
			{
				std::printf("dnan1 %g %g\n", first[i], desiredChanges[i]);
				std::exit(0);
			}
			updatem(mv.first.bm[i], d);
			updatev(mv.first.bv[i], d);
			updateparam(out.first.biases[i], alphat, mv.first.bm[i], mv.first.bv[i]);
			// out.first.biases[i] *= (1.0 - wd);
		}

		for (size_t i = 0; i < f; ++i)
		{
			for (size_t j = 0; j < prevSize; ++j)
			{
				const double d = cderiv<Act>(prev[j], first[i], desiredChanges[i]) + wd * conns.first.weights[i][j];
				if (std::isnan(d))
				{
					std::printf("dnan2 %g %g %g\n", prev[i], first[i], desiredChanges[i]);
					std::exit(0);
				}
				updatem(mv.first.wm[i][j], d);
				updatev(mv.first.wv[i][j], d);
				updateparam(out.first.weights[i][j], alphat, mv.first.wm[i][j], mv.first.wv[i][j]);
				// out.first.weights[i][j] *= (1.0 - wd);
			}
		}
	}

	double getProb(size_t output) const noexcept
	{
		return rest.getProb(output);
	}

	void print(size_t n) const noexcept
	{
		std::printf("Lager %zu:\n", ++n);
		for (const auto& i : first)
		{
			std::printf("%g\n", i);
		}
		rest.print(n);
	}

	static constexpr size_t outputSize() noexcept
	{
		return decltype(rest)::outputSize();
	}
};

template <class Act, class LastAct, size_t f>
struct NeuronLayers<Act, LastAct, f>
{
	Neurons<f> first;

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

	template <size_t prevSize>
	void feedForward(const Neurons<prevSize>& prev, const ConnectionLayers<Act, LastAct, prevSize, f>& conns) noexcept
	{
		first = conns.first.biases;

		for (size_t i = 0; i < first.size(); ++i)
		{
			for (size_t j = 0; j < prev.size(); ++j)
			{
				first[i] = std::clamp(first[i] + prev[j] * conns.first.weights[i][j], -DBL_MAX, DBL_MAX);
				if (std::isnan(first[i]))
				{
					std::puts("NAN-3");
					std::exit(0);
				}
			}
			first[i] = LastAct::act(first[i]);
		}
	}

	template <size_t prevSize, class PatchT>
	void backprop(
		const std::array<double, prevSize>& prev,
		const ConnectionLayers<Act, LastAct, prevSize, f>& /*conns*/,
		const std::array<double, f>& desiredChanges,
		double wd,
		Patch<PatchT>& out
	) const
	{
		for (size_t i = 0; i < f; ++i)
		{
			out.first.biases[i] = std::clamp(out.first.biases[i] + cpderiv(desiredChanges[i]), -DBL_MAX, DBL_MAX);
			out.first.biases[i] *= wd;
		}

		for (size_t i = 0; i < f; ++i)
		{
			for (size_t j = 0; j < prevSize; ++j)
			{
				out.first.weights[i][j] = std::clamp(out.first.weights[i][j] + cpderiv(prev[j], desiredChanges[i]), -DBL_MAX, DBL_MAX);
				out.first.weights[i][j] *= wd;
			}
		}
	}

	template <size_t prevSize, class PatchT>
	void backprop(
		const std::array<double, prevSize>& prev,
		const ConnectionLayers<Act, LastAct, prevSize, f>& conns,
		const std::array<double, f>& desiredChanges,
		double wd,
		Patch<PatchT>& out,
		std::array<double, prevSize>& nextDesiredChanges
	) const
	{
		for (size_t i = 0; i < f; ++i)
		{
			out.first.biases[i] = std::clamp(out.first.biases[i] + cpderiv(desiredChanges[i]), -DBL_MAX, DBL_MAX);
			out.first.biases[i] *= wd;
		}

		for (size_t i = 0; i < f; ++i)
		{
			for (size_t j = 0; j < prevSize; ++j)
			{
				out.first.weights[i][j] = std::clamp(out.first.weights[i][j] + cpderiv(prev[j], desiredChanges[i]), -DBL_MAX, DBL_MAX);
				out.first.weights[i][j] *= wd;
			}
		}

		// Ja, det är j ytterst.
		for (size_t j = 0; j < prevSize; ++j)
		{
			nextDesiredChanges[j] = 0.0;
			for (size_t i = 0; i < f; ++i)
			{
				nextDesiredChanges[j] = std::clamp(nextDesiredChanges[j] + cpderiv(conns.first.weights[i][j], desiredChanges[i]), -DBL_MAX, DBL_MAX);
			}
		}
	}

	template <size_t prevSize, class PatchT>
	void adam(
		MomentVecs<prevSize, f>& mv,
		size_t t,
		const std::array<double, prevSize>& prev,
		const ConnectionLayers<Act, LastAct, prevSize, f>& conns,
		const std::array<double, f>& desiredChanges,
		double wd,
		Patch<PatchT>& out,
		std::array<double, prevSize>& nextDesiredChanges
	) const
	{
		const double alphat = calcalphat(t);

		for (size_t i = 0; i < f; ++i)
		{
			const double d = cderiv<LastAct>(first[i], desiredChanges[i]) + wd * conns.first.biases[i];
			if (std::isnan(d))
			{
				std::printf("dnan3 %g %g\n", first[i], desiredChanges[i]);
				std::exit(0);
			}
			updatem(mv.first.bm[i], d);
			updatev(mv.first.bv[i], d);
			updateparam(out.first.biases[i], alphat, mv.first.bm[i], mv.first.bv[i]);
			// out.first.biases[i] *= (1.0 - wd);
		}

		for (size_t i = 0; i < f; ++i)
		{
			for (size_t j = 0; j < prevSize; ++j)
			{
				const double d = cderiv<LastAct>(prev[j], first[i], desiredChanges[i]) + wd * conns.first.weights[i][j];
				if (std::isnan(d))
				{
					std::printf("dnan4 %g %g %g\n", prev[i], first[i], desiredChanges[i]);
					std::exit(0);
				}
				updatem(mv.first.wm[i][j], d);
				updatev(mv.first.wv[i][j], d);
				updateparam(out.first.weights[i][j], alphat, mv.first.wm[i][j], mv.first.wv[i][j]);
				// out.first.weights[i][j] *= (1.0 - wd);
			}
		}

		// Ja, det är j ytterst.
		for (size_t j = 0; j < prevSize; ++j)
		{
			nextDesiredChanges[j] = 0.0;
			for (size_t i = 0; i < f; ++i)
			{
				nextDesiredChanges[j] = std::clamp(nextDesiredChanges[j] + cderiv<LastAct>(conns.first.weights[i][j], first[i], desiredChanges[i]), -DBL_MAX, DBL_MAX);
			}
		}
	}

	template <size_t prevSize, class PatchT>
	void adam(
		MomentVecs<prevSize, f>& mv,
		size_t t,
		const std::array<double, prevSize>& prev,
		const ConnectionLayers<Act, LastAct, prevSize, f>& conns,
		const std::array<double, f>& desiredChanges,
		double wd,
		Patch<PatchT>& out
	) const
	{
		const double alphat = calcalphat(t);

		for (size_t i = 0; i < f; ++i)
		{
			const double d = cderiv<LastAct>(first[i], desiredChanges[i]) + wd * conns.first.biases[i];
			if (std::isnan(d))
			{
				std::printf("dnan3 %g %g\n", first[i], desiredChanges[i]);
				std::exit(0);
			}
			updatem(mv.first.bm[i], d);
			updatev(mv.first.bv[i], d);
			updateparam(out.first.biases[i], alphat, mv.first.bm[i], mv.first.bv[i]);
			// out.first.biases[i] *= (1.0 - wd);
		}

		for (size_t i = 0; i < f; ++i)
		{
			for (size_t j = 0; j < prevSize; ++j)
			{
				const double d = cderiv<LastAct>(prev[j], first[i], desiredChanges[i]) + wd * conns.first.weights[i][j];
				if (std::isnan(d))
				{
					std::printf("dnan4 %g %g %g\n", prev[i], first[i], desiredChanges[i]);
					std::exit(0);
				}
				updatem(mv.first.wm[i][j], d);
				updatev(mv.first.wv[i][j], d);
				updateparam(out.first.weights[i][j], alphat, mv.first.wm[i][j], mv.first.wv[i][j]);
				// out.first.weights[i][j] *= (1.0 - wd);
			}
		}
	}

	double getProb(size_t output) const noexcept
	{
		return first[output];
	}

	void print(size_t n) const noexcept
	{
		std::printf("Lager %zu:\n", ++n);
		for (const auto& i : first)
		{
			std::printf("%g\n", i);
		}
	}

	static constexpr size_t outputSize() noexcept
	{
		return f;
	}
};