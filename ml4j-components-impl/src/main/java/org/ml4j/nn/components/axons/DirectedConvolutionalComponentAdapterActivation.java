package org.ml4j.nn.components.axons;

import org.ml4j.nn.components.DirectedComponentActivation;
import org.ml4j.nn.components.DirectedComponentGradient;
import org.ml4j.nn.neurons.NeuronsActivation;

public class DirectedConvolutionalComponentAdapterActivation implements DirectedComponentActivation<NeuronsActivation, NeuronsActivation> {

	@Override
	public DirectedComponentGradient<NeuronsActivation> backPropagate(
			DirectedComponentGradient<NeuronsActivation> gradient) {
		return gradient;
	}

	@Override
	public NeuronsActivation getOutput() {
		// TODO Auto-generated method stub
		return null;
	}

}
