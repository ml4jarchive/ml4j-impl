package org.ml4j.nn.components.onetoone.base;

import org.ml4j.nn.components.base.DefaultChainableDirectedComponentActivationBase;
import org.ml4j.nn.components.onetone.DefaultDirectedComponentChainActivation;
import org.ml4j.nn.neurons.NeuronsActivation;

public abstract class DefaultDirectedComponentChainActivationBase extends DefaultChainableDirectedComponentActivationBase implements DefaultDirectedComponentChainActivation {
	
	public DefaultDirectedComponentChainActivationBase(NeuronsActivation output) {
		super(output);
	}
	
}
