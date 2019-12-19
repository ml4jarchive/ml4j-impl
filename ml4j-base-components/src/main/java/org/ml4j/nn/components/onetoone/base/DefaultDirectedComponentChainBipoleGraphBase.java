package org.ml4j.nn.components.onetoone.base;

import org.ml4j.nn.components.DirectedComponentType;
import org.ml4j.nn.components.DirectedComponentsContext;
import org.ml4j.nn.components.manytomany.DefaultDirectedComponentChainBatch;
import org.ml4j.nn.components.onetone.DefaultDirectedComponentBipoleGraph;
import org.ml4j.nn.components.onetone.DefaultDirectedComponentChain;
import org.ml4j.nn.components.onetone.DefaultDirectedComponentChainActivation;

public abstract class DefaultDirectedComponentChainBipoleGraphBase implements DefaultDirectedComponentBipoleGraph {

	/**
	 * Default serialization id.
	 */
	private static final long serialVersionUID = 1L;
	
	private DefaultDirectedComponentChainBatch<DefaultDirectedComponentChain, DefaultDirectedComponentChainActivation> parallelComponentChainsBatch;

	public DefaultDirectedComponentChainBipoleGraphBase(
			DefaultDirectedComponentChainBatch<DefaultDirectedComponentChain, DefaultDirectedComponentChainActivation> parallelComponentChainsBatch) {
		this.parallelComponentChainsBatch = parallelComponentChainsBatch;
	}
	
	@Override
	public DirectedComponentsContext getContext(DirectedComponentsContext context, int componentIndex) {
		return context;
	}
	@Override
	public DirectedComponentType getComponentType() {
		return DirectedComponentType.COMPONENT_CHAIN_GRAPH;
	}
}
