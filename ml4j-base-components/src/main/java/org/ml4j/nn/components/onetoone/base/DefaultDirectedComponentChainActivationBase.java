package org.ml4j.nn.components.onetoone.base;

import java.util.List;

import org.ml4j.nn.components.base.DefaultChainableDirectedComponentActivationBase;
import org.ml4j.nn.components.onetone.DefaultChainableDirectedComponent;
import org.ml4j.nn.components.onetone.DefaultChainableDirectedComponentActivation;
import org.ml4j.nn.components.onetone.DefaultDirectedComponentChainActivation;
import org.ml4j.nn.neurons.NeuronsActivation;

/**
 * Default base class for implementations of DefaultDirectedComponentChainActivation.
 * 
 * Encapsulates the activations from a forward propagation through a DefaultDirectedComponentChain.
 * 
 * @author Michael Lavelle
 */
public abstract class DefaultDirectedComponentChainActivationBase<L extends DefaultChainableDirectedComponent<?, ?>> extends DefaultChainableDirectedComponentActivationBase<L> implements DefaultDirectedComponentChainActivation {
	
	protected List<DefaultChainableDirectedComponentActivation> activations;
	
	public DefaultDirectedComponentChainActivationBase(L componentChain,  List<DefaultChainableDirectedComponentActivation> activations, NeuronsActivation output) {
		super(componentChain, output);
		this.activations = activations;
	}

	@Override
	public List<DefaultChainableDirectedComponentActivation> getActivations() {
		return activations;
	}
	
	
	
}
