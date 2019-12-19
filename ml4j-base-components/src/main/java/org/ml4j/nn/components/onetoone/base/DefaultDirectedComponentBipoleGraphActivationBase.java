package org.ml4j.nn.components.onetoone.base;

import org.ml4j.nn.components.base.DefaultChainableDirectedComponentActivationBase;
import org.ml4j.nn.components.onetone.DefaultDirectedComponentBipoleGraphActivation;
import org.ml4j.nn.neurons.NeuronsActivation;

public abstract class DefaultDirectedComponentBipoleGraphActivationBase extends DefaultChainableDirectedComponentActivationBase
		implements DefaultDirectedComponentBipoleGraphActivation {
	
	public DefaultDirectedComponentBipoleGraphActivationBase(NeuronsActivation output) {
		super(output);
	}

}
