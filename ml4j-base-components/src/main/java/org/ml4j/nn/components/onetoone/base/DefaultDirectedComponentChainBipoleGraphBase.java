package org.ml4j.nn.components.onetoone.base;

import org.ml4j.nn.components.DirectedComponentType;
import org.ml4j.nn.components.DirectedComponentsContext;
import org.ml4j.nn.components.manytomany.DefaultDirectedComponentChainBatch;
import org.ml4j.nn.components.onetone.DefaultDirectedComponentBipoleGraph;
import org.ml4j.nn.components.onetone.DefaultDirectedComponentChain;
import org.ml4j.nn.components.onetone.DefaultDirectedComponentChainActivation;
import org.ml4j.nn.neurons.Neurons;

public abstract class DefaultDirectedComponentChainBipoleGraphBase implements DefaultDirectedComponentBipoleGraph {

	/**
	 * Default serialization id.
	 */
	private static final long serialVersionUID = 1L;
	
	protected DefaultDirectedComponentChainBatch<DefaultDirectedComponentChain, DefaultDirectedComponentChainActivation> parallelComponentChainsBatch;

	protected Neurons inputNeurons;
	protected Neurons outputNeurons;
	
	public DefaultDirectedComponentChainBipoleGraphBase(Neurons inputNeurons, Neurons outputNeurons,
			DefaultDirectedComponentChainBatch<DefaultDirectedComponentChain, DefaultDirectedComponentChainActivation> parallelComponentChainsBatch) {
		this.parallelComponentChainsBatch = parallelComponentChainsBatch;
		this.inputNeurons = inputNeurons;
		this.outputNeurons = outputNeurons;
	}
	
	@Override
	public DirectedComponentsContext getContext(DirectedComponentsContext context, int componentIndex) {
		return context;
	}
	@Override
	public DirectedComponentType getComponentType() {
		return DirectedComponentType.COMPONENT_CHAIN_GRAPH;
	}

	@Override
	public Neurons getInputNeurons() {
		return inputNeurons;
	}

	@Override
	public Neurons getOutputNeurons() {
		return outputNeurons;
	}

}
