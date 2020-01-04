package org.ml4j.nn.components.axons.base;

import org.ml4j.nn.axons.Axons;
import org.ml4j.nn.axons.AxonsContext;
import org.ml4j.nn.components.axons.DirectedAxonsComponent;
import org.ml4j.nn.components.axons.DirectedAxonsComponentActivation;
import org.ml4j.nn.components.onetoone.DefaultChainableDirectedComponentAdapter;
import org.ml4j.nn.neurons.Neurons;
import org.ml4j.nn.neurons.NeuronsActivation;

public class DirectedAxonsComponentAdapter<L extends Neurons, R extends Neurons> extends DefaultChainableDirectedComponentAdapter<DirectedAxonsComponentActivation, AxonsContext> 
	implements DirectedAxonsComponent<L, R, Axons<L, R, ?>> {

	/**
	 * Default serialization id.
	 */
	private static final long serialVersionUID = 1L;

	public DirectedAxonsComponentAdapter(
			DirectedAxonsComponent<L, R, ?> delegated) {
		super(delegated, delegated.getClass().getSimpleName() + ":" + delegated.getAxons().getClass().getSimpleName());
	}

	@SuppressWarnings("unchecked")
	@Override
	public Axons<L, R, ?> getAxons() {
		return (Axons<L, R, ?>)((DirectedAxonsComponent<L, R, ?>)delegated).getAxons();
	}

	@SuppressWarnings("unchecked")
	@Override
	public DirectedAxonsComponentAdapter<L, R> dup() {
		return new DirectedAxonsComponentAdapter<L, R>((DirectedAxonsComponent<L, R, ?>) delegated.dup());
	}

	@Override
	public DirectedAxonsComponentActivation forwardPropagate(NeuronsActivation input, AxonsContext context) {
		return new DirectedAxonsComponentActivationAdapter(super.forwardPropagate(input, context), name);
	}
	
	

}
