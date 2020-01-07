package org.ml4j.nn.components.onetoone.base;

import org.ml4j.nn.components.base.DefaultChainableDirectedComponentActivationBase;
import org.ml4j.nn.components.onetone.DefaultDirectedComponentBipoleGraph;
import org.ml4j.nn.components.onetone.DefaultDirectedComponentBipoleGraphActivation;
import org.ml4j.nn.neurons.NeuronsActivation;

/**
 * Default base class for an activation from a DefaultDirectedComponentBipoleGraph.
 * 
 * Encapsulates the activations from a forward propagation through a DefaultDirectedComponentBipoleGraph including the
 * output NeuronsActivation from the RHS of the DefaultDirectedComponentBipoleGraph.
 * 
 * @author Michael Lavelle
 *
 */
public abstract class DefaultDirectedComponentBipoleGraphActivationBase extends DefaultChainableDirectedComponentActivationBase<DefaultDirectedComponentBipoleGraph>
		implements DefaultDirectedComponentBipoleGraphActivation {
	
	public DefaultDirectedComponentBipoleGraphActivationBase(DefaultDirectedComponentBipoleGraph bipoleGraph, NeuronsActivation output) {
		super(bipoleGraph, output);
	}

}
