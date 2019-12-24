package org.ml4j.nn.activationfunctions.mocks;

import org.ml4j.nn.activationfunctions.DifferentiableActivationFunction;
import org.ml4j.nn.activationfunctions.DifferentiableActivationFunctionActivation;
import org.ml4j.nn.neurons.NeuronsActivation;

public class DummyDifferentiableActivationFunctionActivation implements DifferentiableActivationFunctionActivation {

	private NeuronsActivation output;
	private NeuronsActivation input;
	private DifferentiableActivationFunction activationFunction;
	
	public DummyDifferentiableActivationFunctionActivation(DifferentiableActivationFunction activationFunction, NeuronsActivation input, NeuronsActivation output) {
		this.output = output;
		this.input = input;
		this.activationFunction = activationFunction;	
	}
	
	
	@Override
	public DifferentiableActivationFunction getActivationFunction() {
		return activationFunction;
	}

	@Override
	public NeuronsActivation getInput() {
		return input;
	}

	@Override
	public NeuronsActivation getOutput() {
		return output;
	}

}
