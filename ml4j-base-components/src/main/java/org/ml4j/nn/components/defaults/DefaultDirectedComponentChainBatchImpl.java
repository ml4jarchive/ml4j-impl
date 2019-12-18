package org.ml4j.nn.components.defaults;

import java.util.List;
import java.util.stream.Collectors;

import org.ml4j.nn.components.ChainableDirectedComponentActivation;
import org.ml4j.nn.components.DirectedComponentBatchActivation;
import org.ml4j.nn.components.DirectedComponentBatchImpl;
import org.ml4j.nn.components.DirectedComponentChain;
import org.ml4j.nn.components.DirectedComponentChainActivation;
import org.ml4j.nn.components.DirectedComponentChainBatch;
import org.ml4j.nn.components.DirectedComponentType;
import org.ml4j.nn.components.DirectedComponentsContext;
import org.ml4j.nn.neurons.NeuronsActivation;

public class DefaultDirectedComponentChainBatchImpl<L extends DirectedComponentChain<NeuronsActivation, ?, ? , T>, T extends DirectedComponentChainActivation<NeuronsActivation, A >, A extends ChainableDirectedComponentActivation<NeuronsActivation>> extends DirectedComponentBatchImpl<NeuronsActivation, L, T, DirectedComponentsContext, DirectedComponentsContext> 
	implements DirectedComponentChainBatch<NeuronsActivation, L, T, DirectedComponentBatchActivation<NeuronsActivation, T>>,  DefaultDirectedComponentChainBatch<L, T> {

	
	/**
	 * Defualt serialziation id.
	 */
	private static final long serialVersionUID = 1L;

	public DefaultDirectedComponentChainBatchImpl(List<L> components) {
		super(components);
	}

	@Override
	protected DirectedComponentsContext getContext(DirectedComponentsContext context, L component, int index) {
		return context;
	}

	@SuppressWarnings("unchecked")
	@Override
	public DefaultDirectedComponentChainBatch<L, T> dup() {
		return new DefaultDirectedComponentChainBatchImpl<L, T, A>((List<L>) getComponents().stream().map(c -> c.dup()).collect(Collectors.toList()));
	}

	@Override
	public DirectedComponentType getComponentType() {
		return DirectedComponentType.COMPONENT_CHAIN_BATCH;
	}

	
}
