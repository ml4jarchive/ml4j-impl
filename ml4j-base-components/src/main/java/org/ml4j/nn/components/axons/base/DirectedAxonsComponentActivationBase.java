package org.ml4j.nn.components.axons.base;

import java.util.Arrays;
import java.util.List;

import org.ml4j.nn.axons.Axons;
import org.ml4j.nn.components.axons.DirectedAxonsComponent;
import org.ml4j.nn.components.axons.DirectedAxonsComponentActivation;
import org.ml4j.nn.components.base.DefaultChainableDirectedComponentActivationBase;
import org.ml4j.nn.components.onetone.DefaultChainableDirectedComponentActivation;
import org.ml4j.nn.neurons.NeuronsActivation;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Default base class for implementations of DirectedAxonsComponentActivation.
 * 
 * Encapsulates the activations from a forward propagation through a DirectedAxonsComponent
 * 
 * @author Michael Lavelle
 */
public abstract class DirectedAxonsComponentActivationBase<A extends Axons<?, ?, ?>> extends DefaultChainableDirectedComponentActivationBase<DirectedAxonsComponent<?, ?, A>> implements DirectedAxonsComponentActivation {
	
	@SuppressWarnings("unused")
	private static final Logger LOGGER = LoggerFactory.getLogger(DirectedAxonsComponentActivationBase.class);
	
	/**
	 * The DirectedAxonsComponent that generated this activation.
	 */
	protected DirectedAxonsComponent<?, ?, A> directedAxonsComponent;

	/**
	 * Constructor for DirectedAxonsComponentActivationBase
	 * 
	 * @param axonsComponent The DirectedAxonsComponent that generated this activation.
	 * @param output The NeuronsActivation output on the RHS of the forward propagation.
	 */
	public DirectedAxonsComponentActivationBase(DirectedAxonsComponent<?, ?, A> axonsComponent, NeuronsActivation output) {
		super(axonsComponent, output);
		this.directedAxonsComponent = axonsComponent;
	}
	
	@Override
	public List<DefaultChainableDirectedComponentActivation> decompose() {
		// By default, a DirectedAxonsComponentActivation cannot be decomposed into smaller components, so return a singleton list containing this component.
		return Arrays.asList(this);
	}

	@Override
	public double getAverageRegularisationCost() {
		return getTotalRegularisationCost() / output.getExampleCount();
	}

	@Override
	public DirectedAxonsComponent<?, ?, A> getAxonsComponent() {
		return directedAxonsComponent;
	}


}
