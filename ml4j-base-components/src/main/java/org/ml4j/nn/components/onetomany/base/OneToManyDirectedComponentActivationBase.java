package org.ml4j.nn.components.onetomany.base;

import java.util.ArrayList;
import java.util.List;

import org.ml4j.nn.components.onetomany.OneToManyDirectedComponentActivation;
import org.ml4j.nn.neurons.NeuronsActivation;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public abstract class OneToManyDirectedComponentActivationBase implements OneToManyDirectedComponentActivation {

	private static final Logger LOGGER = LoggerFactory.getLogger(OneToManyDirectedComponentActivationBase.class);
	
	private NeuronsActivation input;
	private int size;
	
	public OneToManyDirectedComponentActivationBase(NeuronsActivation input) {
		this.input = input;
	}

	@Override
	public List<NeuronsActivation> getOutput() {
		List<NeuronsActivation> outputs = new ArrayList<>();
		for (int i = 0; i < size; i++) {
			outputs.add(input);
		}
		return outputs;
	}

}
