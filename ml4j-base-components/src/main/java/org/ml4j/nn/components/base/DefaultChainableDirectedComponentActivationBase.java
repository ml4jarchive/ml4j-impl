package org.ml4j.nn.components.base;

import org.ml4j.nn.components.onetone.DefaultChainableDirectedComponent;
import org.ml4j.nn.components.onetone.DefaultChainableDirectedComponentActivation;
import org.ml4j.nn.neurons.NeuronsActivation;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Default base class for implementations of DefaultChainableDirectedComponentActivation.
 * 
 * Encapsulates the activations from a forward propagation through a ChainableDirectedComponent
 * 
 * 
 * @author Michael Lavelle
 */
public abstract class DefaultChainableDirectedComponentActivationBase<L extends DefaultChainableDirectedComponent<?, ?>> implements DefaultChainableDirectedComponentActivation {
	
	@SuppressWarnings("unused")
	private static final Logger LOGGER = LoggerFactory.getLogger(DefaultChainableDirectedComponentActivationBase.class);
	
	/**
	 * The NeuronsActivation output on the RHS of the forward propagation.
	 */
	protected NeuronsActivation output;
	
	protected L originatingComponent;

	/**
	 * @param originatingComponent The component from which this activation originated
	 * @param output The NeuronsActivation output on the RHS of the forward propagation.
	 */
	public DefaultChainableDirectedComponentActivationBase(L originatingComponent, NeuronsActivation output) {
		this.output = output;
		this.originatingComponent = originatingComponent;
	}

	@Override
	public NeuronsActivation getOutput() {
		return output;
	}
}


