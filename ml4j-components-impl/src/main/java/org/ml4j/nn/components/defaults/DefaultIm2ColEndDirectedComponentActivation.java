package org.ml4j.nn.components.defaults;

import java.util.Arrays;
import java.util.List;

import org.ml4j.nn.components.ChainableDirectedComponentActivation;
import org.ml4j.nn.components.DirectedComponentGradient;
import org.ml4j.nn.neurons.ImageNeuronsActivation;
import org.ml4j.nn.neurons.NeuronsActivation;

public class DefaultIm2ColEndDirectedComponentActivation implements ChainableDirectedComponentActivation<NeuronsActivation> {

	private ImageNeuronsActivation output;
	
	public DefaultIm2ColEndDirectedComponentActivation(ImageNeuronsActivation output) {
		this.output = output;
	}
	
	@Override
	public DirectedComponentGradient<NeuronsActivation> backPropagate(
			DirectedComponentGradient<NeuronsActivation> arg0) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public NeuronsActivation getOutput() {
		return output;
	}

	@Override
	public List<ChainableDirectedComponentActivation<NeuronsActivation>> decompose() {
		return Arrays.asList(this);
	}

}
