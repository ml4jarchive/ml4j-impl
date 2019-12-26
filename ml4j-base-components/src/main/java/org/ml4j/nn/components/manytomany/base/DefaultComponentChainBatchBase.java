package org.ml4j.nn.components.manytomany.base;

import java.util.List;

import org.ml4j.nn.components.DirectedComponentType;
import org.ml4j.nn.components.manytomany.DefaultDirectedComponentChainBatch;
import org.ml4j.nn.components.onetone.DefaultDirectedComponentChain;

public abstract class DefaultComponentChainBatchBase implements DefaultDirectedComponentChainBatch {

	/**
	 * Default serialization id.
	 */
	private static final long serialVersionUID = 1L;
	
	protected List<DefaultDirectedComponentChain> parallelComponents;

	public DefaultComponentChainBatchBase(List<DefaultDirectedComponentChain> parallelComponents) {
		this.parallelComponents = parallelComponents;
	}

	@Override
	public List<DefaultDirectedComponentChain> getComponents() {
		return parallelComponents;
	}

	@Override
	public DirectedComponentType getComponentType() {
		return DirectedComponentType.COMPONENT_CHAIN_BATCH;
	}

}
