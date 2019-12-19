package org.ml4j.nn.components.manytoone.base;

import org.ml4j.nn.components.manytoone.ManyToOneDirectedComponentActivation;
import org.ml4j.nn.neurons.NeuronsActivation;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public abstract class ManyToOneDirectedComponentActivationBase implements ManyToOneDirectedComponentActivation {

	private static final Logger LOGGER = LoggerFactory.getLogger(ManyToOneDirectedComponentActivationBase.class);
	
	protected NeuronsActivation output;
	
	public ManyToOneDirectedComponentActivationBase(NeuronsActivation output) {
		this.output = output;
	}

	@Override
	public NeuronsActivation getOutput() {
		return output;
	}

}
