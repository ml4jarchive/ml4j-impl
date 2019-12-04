package org.ml4j.nn.components.builders.base;

import org.ml4j.Matrix;
import org.ml4j.nn.components.builders.BaseGraphBuilderState;
import org.ml4j.nn.components.builders.axons.UncompletedFullyConnectedAxonsBuilder;
import org.ml4j.nn.components.builders.componentsgraph.ComponentsGraphNeurons;
import org.ml4j.nn.components.builders.synapses.SynapsesAxonsGraphBuilder;
import org.ml4j.nn.neurons.Neurons;

public class BaseGraphBuilderStateImpl implements BaseGraphBuilderState {
	
	protected ComponentsGraphNeurons<Neurons> componentsGraphNeurons;
	protected Matrix connectionWeights;
	protected Matrix biases;
	protected UncompletedFullyConnectedAxonsBuilder<?> fullyConnectedAxonsBuilder;
	protected SynapsesAxonsGraphBuilder<?> synapsesBuilder;
	
	public BaseGraphBuilderStateImpl(Neurons initialNeurons) {
		this.componentsGraphNeurons = new ComponentsGraphNeuronsImpl<>(initialNeurons);
	}
	
	public BaseGraphBuilderStateImpl() {
	}
	
	@Override
	public ComponentsGraphNeurons<Neurons> getComponentsGraphNeurons() {
		return componentsGraphNeurons;
	}
	
	public void setComponentsGraphNeurons(ComponentsGraphNeurons<Neurons> componentsGraphNeurons) {
		this.componentsGraphNeurons = componentsGraphNeurons;
	}
	
	@Override
	public Matrix getConnectionWeights() {
		return connectionWeights;
	}
	
	@Override
	public void setConnectionWeights(Matrix connectionWeights) {
		this.connectionWeights = connectionWeights;
	}
	
	@Override
	public Matrix getBiases() {
		return biases;
	}

	@Override
	public void setBiases(Matrix biases) {
		this.biases = biases;
	}

	@Override
	public UncompletedFullyConnectedAxonsBuilder<?> getFullyConnectedAxonsBuilder() {
		return fullyConnectedAxonsBuilder;
	}
	
	@Override
	public void setFullyConnectedAxonsBuilder(UncompletedFullyConnectedAxonsBuilder<?> fullyConnectedAxonsBuilder) {
		this.fullyConnectedAxonsBuilder = fullyConnectedAxonsBuilder;
	}
	
	@Override
	public SynapsesAxonsGraphBuilder<?> getSynapsesBuilder() {
		return synapsesBuilder;
	}
	
	@Override
	public void setSynapsesBuilder(SynapsesAxonsGraphBuilder<?> synapsesBuilder) {
		this.synapsesBuilder = synapsesBuilder;
	}
}
