package org.ml4j.nn.activationfunctions.mocks;

import java.util.Arrays;
import java.util.List;

import org.ml4j.nn.activationfunctions.DifferentiableActivationFunction;
import org.ml4j.nn.activationfunctions.DifferentiableActivationFunctionActivation;
import org.ml4j.nn.components.DirectedComponentGradient;
import org.ml4j.nn.components.DirectedComponentGradientImpl;
import org.ml4j.nn.components.onetone.DefaultChainableDirectedComponentActivation;
import org.ml4j.nn.costfunctions.CostFunctionGradient;
import org.ml4j.nn.neurons.NeuronsActivation;

public class DummyDifferentiableActivationFunctionActivation implements DifferentiableActivationFunctionActivation {

	private NeuronsActivation output;
	
	public DummyDifferentiableActivationFunctionActivation(NeuronsActivation output) {
		this.output = output;
	}
	
	
	@Override
	public DifferentiableActivationFunction getActivationFunction() {
		throw new UnsupportedOperationException();
	}

	@Override
	public NeuronsActivation getInput() {
		throw new UnsupportedOperationException();
	}

	@Override
	public NeuronsActivation getOutput() {
		return output;
	}

	@Override
	public List<DefaultChainableDirectedComponentActivation> decompose() {
		return Arrays.asList(this);
	}

	@Override
	public DirectedComponentGradient<NeuronsActivation> backPropagate(
			DirectedComponentGradient<NeuronsActivation> gradient) {
		return gradient;
	}

	@Override
	public DirectedComponentGradient<NeuronsActivation> backPropagate(CostFunctionGradient costFunctionGradient) {
		return new DirectedComponentGradientImpl<>(output);
	}

}
