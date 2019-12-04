package org.ml4j.nn.components.axons;

import java.util.List;

import org.ml4j.nn.components.DirectedComponentChainActivation;
import org.ml4j.nn.components.defaults.DefaultDirectedComponentChainBatchImpl;
import org.ml4j.nn.neurons.Neurons;
import org.ml4j.nn.neurons.NeuronsActivation;

public class DirectedAxonsComponentChainBatchImpl<L extends Neurons, R extends Neurons> extends DefaultDirectedComponentChainBatchImpl<DirectedAxonsComponentChain<L, R>, DirectedComponentChainActivation<NeuronsActivation, DirectedAxonsComponentActivation>, DirectedAxonsComponentActivation>
 implements DirectedAxonsComponentChainBatch<L, R> {

	/**
	 * Default serialization id.
	 */
	private static final long serialVersionUID = 1L;

	public DirectedAxonsComponentChainBatchImpl(List<DirectedAxonsComponentChain<L, R>> components) {
		super(components);
	}

}
