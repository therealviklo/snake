#pragma once
#include <array>
#include <cmath>
#include <filesystem>
#include "fio.h"
#include "game.h"
#include "rand.h"
#include "neurons.h"
#include "patch.h"
#include "activation.h"
#include "adam.h"

template <class Act, class LastAct, size_t inputSize, size_t ...sizes>
struct Layers
{
	NeuronLayers<Act, LastAct, sizes...> neurons;
	ConnectionLayers<Act, LastAct, inputSize, sizes...> conns;

	void feedForward(const Neurons<inputSize>& inputs) noexcept
	{
		neurons.feedForward(inputs, conns);
	}

	template <size_t lastSize, class PatchT>
	void backprop(const Neurons<inputSize>& inputs, const std::array<double, lastSize>& inputDiff, double wd, Patch<PatchT>& out) const
	{
		neurons.backprop(
			inputs,
			conns,
			inputDiff,
			wd,
			out
		);
	}

	template <size_t lastSize, class PatchT>
	void adam(Adam<inputSize, sizes...>& a, const Neurons<inputSize>& inputs, const std::array<double, lastSize>& inputDiff, double wd, Patch<PatchT>& out) const
	{
		neurons.adam(
			a.mv,
			a.t++,
			inputs,
			conns,
			inputDiff,
			wd,
			out
		);
	}

	template <class PatchT>
	void applyPatch(const Patch<PatchT>& patch)
	{
		conns.applyPatch(patch);
	}

	double getProb(size_t output) const noexcept
	{
		return neurons.getProb(output);
	}

	static constexpr size_t outputSize() noexcept 
	{
		return decltype(neurons)::outputSize();
	}
};

template <class InputsType, class NeuronsType>
struct State
{
	InputsType inputs;
	NeuronsType neurons;
};

template <class Act, class LastAct, size_t inputSize, size_t ...sizes>
class Net
{
public:
	Neurons<inputSize> inputs;
private:
	Layers<Act, LastAct, inputSize, sizes...> layers;
public:
	using Patch = Patch<decltype(layers.conns)>;
	using State = State<decltype(inputs), decltype(layers.neurons)>;
	using Conns = decltype(layers.conns);
	using Adam = Adam<inputSize, sizes...>;

	void randomise() { layers.conns.randomise(); }

	void run() noexcept { layers.feedForward(inputs); }
	
	double getTotProb() const noexcept
	{
		double tot = 0.0;
		for (size_t i = 0; i < layers.outputSize(); ++i)
		{
			tot += prob(layers.getProb(i));
		}
		return tot;
	}
	double getProb(size_t output) const noexcept { return layers.getProb(output); }
	double getRealProb(size_t output) const noexcept
	{
		return prob(layers.getProb(output)) / getTotProb();
	}
	
	size_t getTopOutput() const noexcept
	{
		size_t num = 0;
		double val = layers.neurons.template l<sizeof...(sizes) - 1>()[0];
		for (size_t i = 1; i < layers.outputSize(); ++i)
		{
			if (layers.neurons.template l<sizeof...(sizes) - 1>()[i] > val)
			{
				num = i;
				val = layers.neurons.template l<sizeof...(sizes) - 1>()[i];
			}
		}
		return num;
	}
	size_t getTopOutput(double randChance) const
	{
		return rand01() < randChance ? randz() % layers.outputSize() : getTopOutput();
	}
	
	size_t getProbOutput() const
	{
		const double selected = randd(0.0, getTotProb());
		double totYet = 0.0;
		for (size_t i = 0; i < layers.outputSize(); ++i)
		{
			totYet += prob(layers.neurons.template l<sizeof...(sizes) - 1>()[i]);
			if (selected <= totYet) return i;
		}
		return 0;
	}
	size_t getProbOutput(double randChance) const
	{
		return rand01() < randChance ? randz() % layers.outputSize() : getProbOutput();
	}

	void backprop(const std::array<double, decltype(layers)::outputSize()>& desiredDiff, Patch& out, double wd = 1.0) const
	{
		layers.backprop(inputs, desiredDiff, wd, out);
	}
	void adam(Adam& a, const std::array<double, decltype(layers)::outputSize()>& desiredDiff, Patch& out, double wd = 0.0) const
	{
		layers.adam(a, inputs, desiredDiff, wd, out);
	}
	void applyPatch(const Patch& patch) { layers.applyPatch(patch); }

	void loadState(const State& s)
	{
		inputs = s.inputs;
		layers.neurons = s.neurons;
	}
	void storeState(State& s) const
	{
		s.inputs = inputs;
		s.neurons = layers.neurons;
	}

	void loadConns(const Conns& conns)
	{
		layers.conns = conns;
	}
	void storeConns(Conns& conns)
	{
		conns = layers.conns;
	}

	void load(const char* fname)
	{
		if (std::filesystem::exists(fname))
			readFromFile(layers.conns, fname);
		else
			randomise();
	}
	void store(const char* fname) const
	{
		writeToFile(layers.conns, fname);
	}

	void print() const noexcept
	{
		std::puts("Vikter:");
		layers.conns.printWeights(0);
		std::puts("Biaser:");
		layers.conns.printBiases(0);
		std::puts("Neuroner:");
		layers.neurons.print(0);
	}
};

using Actor = Net<Relu, Identity, vsize * vsize * 3 + 4, 4, 4>;
using Critic = Net<Sig, Identity, vsize * vsize * 3 + 4, /*16, 16, 16,*/ 1>;
using Evaluator = Net<Relu, Identity, vsize * vsize * 3 + 4, 4>;