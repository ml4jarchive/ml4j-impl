package org.ml4j.nn.synapses;

import java.util.List;
import java.util.stream.Collectors;

import org.ml4j.nn.components.defaults.DefaultDirectedComponentChainBatchImpl;

public class DirectedSynapsesChainBatchImpl<S extends DirectedSynapses<?, ?>> extends DefaultDirectedComponentChainBatchImpl<DirectedSynapsesChain<S>, DirectedSynapsesChainActivation, DirectedSynapsesActivation>
 implements DirectedSynapsesChainBatch<S> {

	/**
	 * Default serialization id.
	 */
	private static final long serialVersionUID = 1L;

	public DirectedSynapsesChainBatchImpl(List<DirectedSynapsesChain<S>> components) {
		super(components);
	}
	
	@Override
	public DirectedSynapsesChainBatchImpl<S> dup() {
		return new DirectedSynapsesChainBatchImpl<S>((List<DirectedSynapsesChain<S>>) getComponents().stream().map(c -> c.dup()).collect(Collectors.toList()));
	}

}
