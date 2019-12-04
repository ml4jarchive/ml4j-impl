package org.ml4j.nn.components.builders.base;

import org.ml4j.Matrix;
import org.ml4j.nn.components.builders.Base3DGraphBuilderState;
import org.ml4j.nn.components.builders.axons.UncompletedBatchNormAxonsBuilder;
import org.ml4j.nn.components.builders.axons.UncompletedConvolutionalAxonsBuilder;
import org.ml4j.nn.components.builders.axons.UncompletedFullyConnectedAxonsBuilder;
import org.ml4j.nn.components.builders.axons.UncompletedPoolingAxonsBuilder;
import org.ml4j.nn.components.builders.componentsgraph.ComponentsGraphNeurons;
import org.ml4j.nn.components.builders.synapses.SynapsesAxons3DGraphBuilder;
import org.ml4j.nn.neurons.Neurons;
import org.ml4j.nn.neurons.Neurons3D;

public class Base3DGraphBuilderStateImpl implements Base3DGraphBuilderState {

	protected ComponentsGraphNeurons<Neurons3D> componentsGraphNeurons;
	protected UncompletedConvolutionalAxonsBuilder<?> convolutionalAxonsBuilder;
	protected UncompletedFullyConnectedAxonsBuilder<?> fullyConnectedAxonsBuilder;

	protected UncompletedPoolingAxonsBuilder<?> maxPoolingAxonsBuilder;
	protected UncompletedPoolingAxonsBuilder<?> averagePoolingAxonsBuilder;
	protected UncompletedBatchNormAxonsBuilder<?> batchNormAxonsBuilder;


	protected SynapsesAxons3DGraphBuilder<?, ?> synapsesBuilder;
	private Matrix connectionWeights;
	private Matrix biases;
	
	public Base3DGraphBuilderStateImpl() {}
	
	public Base3DGraphBuilderStateImpl(Neurons3D currentNeurons) {
		this.componentsGraphNeurons = new ComponentsGraphNeuronsImpl<>(currentNeurons);
	}
	
	@Override
	public  BaseGraphBuilderStateImpl getNon3DBuilderState() {
		BaseGraphBuilderStateImpl builderState = new BaseGraphBuilderStateImpl();
		builderState.setConnectionWeights(connectionWeights);
		ComponentsGraphNeurons<Neurons> non3DcomponentsGraphNeurons = 
				new ComponentsGraphNeuronsImpl<Neurons>(componentsGraphNeurons.getCurrentNeurons(), componentsGraphNeurons.getRightNeurons());
		
		// TODO ML
		//builderState.setSynapsesBuilder(synapsesBuilder);
		builderState.setComponentsGraphNeurons(non3DcomponentsGraphNeurons);
		builderState.setFullyConnectedAxonsBuilder(fullyConnectedAxonsBuilder);
		return builderState;
	}
	
	@Override
	public ComponentsGraphNeurons<Neurons3D> getComponentsGraphNeurons() {
		return componentsGraphNeurons;
	}
	public void setComponentsGraphNeurons(ComponentsGraphNeurons<Neurons3D> componentsGraphNeurons) {
		this.componentsGraphNeurons = componentsGraphNeurons;
	}
	
	@Override
	public UncompletedConvolutionalAxonsBuilder<?> getConvolutionalAxonsBuilder() {
		return convolutionalAxonsBuilder;
	}
	
	@Override
	public void setConvolutionalAxonsBuilder(UncompletedConvolutionalAxonsBuilder<?> convolutionalAxonsBuilder) {
		this.convolutionalAxonsBuilder = convolutionalAxonsBuilder;
	}
	
	@Override
	public UncompletedPoolingAxonsBuilder<?> getMaxPoolingAxonsBuilder() {
		return maxPoolingAxonsBuilder;
	}
	
	@Override
	public void setMaxPoolingAxonsBuilder(UncompletedPoolingAxonsBuilder<?> maxPoolingAxonsBuilder) {
		this.maxPoolingAxonsBuilder = maxPoolingAxonsBuilder;
	}
	
	@Override
	public SynapsesAxons3DGraphBuilder<?, ?> getSynapsesBuilder() {
		return synapsesBuilder;
	}
	
	@Override
	public void setSynapsesBuilder(SynapsesAxons3DGraphBuilder<?, ?> synapsesBuilder) {
		this.synapsesBuilder = synapsesBuilder;
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
	public void setAveragePoolingAxonsBuilder(UncompletedPoolingAxonsBuilder<?> axonsBuilder) {
		this.averagePoolingAxonsBuilder = axonsBuilder;
	}

	@Override
	public UncompletedPoolingAxonsBuilder<?> getAveragePoolingAxonsBuilder() {
		return averagePoolingAxonsBuilder;
	}

	@Override
	public void setBatchNormAxonsBuilder(UncompletedBatchNormAxonsBuilder<?> axonsBuilder) {
		this.batchNormAxonsBuilder = axonsBuilder;
	}

	@Override
	public UncompletedBatchNormAxonsBuilder<?> getBatchNormAxonsBuilder() {
		return batchNormAxonsBuilder;
	}

}
