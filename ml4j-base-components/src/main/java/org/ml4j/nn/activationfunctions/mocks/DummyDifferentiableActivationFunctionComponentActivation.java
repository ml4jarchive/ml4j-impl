package org.ml4j.nn.activationfunctions.mocks;

import java.util.Arrays;
import java.util.List;

import org.ml4j.nn.activationfunctions.DifferentiableActivationFunctionComponentActivation;
import org.ml4j.nn.components.DirectedComponentGradient;
import org.ml4j.nn.components.DirectedComponentGradientImpl;
import org.ml4j.nn.components.onetone.DefaultChainableDirectedComponentActivation;
import org.ml4j.nn.costfunctions.CostFunctionGradient;
import org.ml4j.nn.neurons.NeuronsActivation;

public class DummyDifferentiableActivationFunctionComponentActivation implements DifferentiableActivationFunctionComponentActivation {

	private NeuronsActivation output;
	
	public DummyDifferentiableActivationFunctionComponentActivation(NeuronsActivation output) {
		this.output = output;
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
