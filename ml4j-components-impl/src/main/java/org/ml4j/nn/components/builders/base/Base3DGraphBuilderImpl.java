package org.ml4j.nn.components.builders.base;

import java.util.ArrayList;
import java.util.List;

import org.ml4j.InterrimMatrix;
import org.ml4j.Matrix;
import org.ml4j.nn.activationfunctions.DifferentiableActivationFunction;
import org.ml4j.nn.activationfunctions.DifferentiableActivationFunctionDirectedComponentImpl;
import org.ml4j.nn.axons.AxonsContext;
import org.ml4j.nn.components.ChainableDirectedComponent;
import org.ml4j.nn.components.ChainableDirectedComponentActivation;
import org.ml4j.nn.components.DirectedComponentChain;
import org.ml4j.nn.components.axons.DirectedAxonsComponent;
import org.ml4j.nn.components.axons.DirectedAxonsComponentFactory;
import org.ml4j.nn.components.builders.Base3DGraphBuilderState;
import org.ml4j.nn.components.builders.axons.Axons3DBuilder;
import org.ml4j.nn.components.builders.axons.Axons3DPermitted;
import org.ml4j.nn.components.builders.axons.AxonsBuilder;
import org.ml4j.nn.components.builders.axons.UncompletedBatchNormAxonsBuilder;
import org.ml4j.nn.components.builders.axons.UncompletedBatchNormAxonsBuilderImpl;
import org.ml4j.nn.components.builders.axons.UncompletedConvolutionalAxonsBuilder;
import org.ml4j.nn.components.builders.axons.UncompletedConvolutionalAxonsBuilderImpl;
import org.ml4j.nn.components.builders.axons.UncompletedFullyConnectedAxonsBuilder;
import org.ml4j.nn.components.builders.axons.UncompletedFullyConnectedAxonsBuilderImpl;
import org.ml4j.nn.components.builders.axons.UncompletedMaxPoolingAxonsBuilderImpl;
import org.ml4j.nn.components.builders.axons.UncompletedPoolingAxonsBuilder;
import org.ml4j.nn.components.builders.common.ComponentsContainer;
import org.ml4j.nn.components.builders.componentsgraph.ComponentsGraphNeurons;
import org.ml4j.nn.components.builders.synapses.Synapses3DPermitted;
import org.ml4j.nn.components.builders.synapses.SynapsesAxons3DGraphBuilder;
import org.ml4j.nn.components.builders.synapses.SynapsesAxons3DGraphBuilderImpl;
import org.ml4j.nn.components.defaults.DefaultDirectedComponentChain;
import org.ml4j.nn.components.defaults.DefaultDirectedComponentChainImpl;
import org.ml4j.nn.neurons.Neurons;
import org.ml4j.nn.neurons.Neurons3D;
import org.ml4j.nn.neurons.NeuronsActivation;

public abstract class Base3DGraphBuilderImpl<C extends Axons3DBuilder, D extends AxonsBuilder> implements Axons3DPermitted<C, D>, Synapses3DPermitted<C, D>, Axons3DBuilder {

	protected DirectedAxonsComponentFactory directedAxonsComponentFactory;
	
	protected Base3DGraphBuilderState initialBuilderState;
	
	private List<ChainableDirectedComponent<NeuronsActivation, ? extends ChainableDirectedComponentActivation<NeuronsActivation>, ?>> components;
	
	private List<DefaultDirectedComponentChain<ChainableDirectedComponentActivation<NeuronsActivation>>> chains;
	private List<Neurons3D> endNeurons;
 
	protected Base3DGraphBuilderState builderState;
		
	public abstract C get3DBuilder();
	public abstract D getBuilder();

	public Base3DGraphBuilderState getBuilderState() {
		return builderState;
	}
	public Matrix getConnectionWeights() {
		return builderState.getConnectionWeights();
	}
	
	public Axons3DBuilder withConnectionWeights(InterrimMatrix connectionWeights) {
		builderState.setConnectionWeights(connectionWeights);
		return this;
	}
	
	public Matrix getBiases() {
		return builderState.getBiases();
	}
	
	public Axons3DBuilder withBiases(InterrimMatrix biases) {
		builderState.setBiases(biases);
		return this;
	}
	
	@Override
	public ComponentsContainer<Neurons> getAxonsBuilder() {
		return getBuilder();
	}
	@Override
	public List<ChainableDirectedComponent<NeuronsActivation, ? extends ChainableDirectedComponentActivation<NeuronsActivation>, ?>> getComponents() {
		addAxonsIfApplicable();
		return components;
	}
	
	@Override
	public ComponentsGraphNeurons<Neurons3D> getComponentsGraphNeurons() {
		return builderState.getComponentsGraphNeurons();
	}

	public Base3DGraphBuilderImpl(DirectedAxonsComponentFactory directedAxonsComponentFactory, Base3DGraphBuilderState builderState, List<ChainableDirectedComponent<NeuronsActivation, ? extends ChainableDirectedComponentActivation<NeuronsActivation>, ?>> components) {
		this.components = components;
		this.directedAxonsComponentFactory = directedAxonsComponentFactory;
		this.builderState = builderState;
		// TODO ML
		this.initialBuilderState = new 	Base3DGraphBuilderStateImpl(builderState.getComponentsGraphNeurons().getCurrentNeurons());
		this.endNeurons = new ArrayList<>();
		this.chains = new ArrayList<>();

	}
	
	public List<DefaultDirectedComponentChain<ChainableDirectedComponentActivation<NeuronsActivation>>> getChains() {
		return chains;
	}
	public List<Neurons3D> getEndNeurons() {
		return endNeurons;
	}
	public void addAxonsIfApplicable() {
		
		if ((builderState.getConvolutionalAxonsBuilder() != null) && builderState.getComponentsGraphNeurons().getRightNeurons() != null) {
			Neurons3D leftNeurons = builderState.getComponentsGraphNeurons().getCurrentNeurons();
			if (builderState.getComponentsGraphNeurons().hasBiasUnit() && !leftNeurons.hasBiasUnit()) {
				leftNeurons = new Neurons3D(builderState.getComponentsGraphNeurons().getCurrentNeurons().getWidth(), builderState.getComponentsGraphNeurons().getCurrentNeurons().getHeight(), builderState.getComponentsGraphNeurons().getCurrentNeurons().getDepth(), true);
			}
			
			DirectedAxonsComponent<Neurons3D, Neurons3D> axonsComponent
					 = directedAxonsComponentFactory.createConvolutionalAxonsComponent(leftNeurons, builderState.getComponentsGraphNeurons().getRightNeurons(), builderState.getConvolutionalAxonsBuilder().getStrideWidth(), builderState.getConvolutionalAxonsBuilder().getStrideHeight(), builderState.getConvolutionalAxonsBuilder().getPaddingWidth(), builderState.getConvolutionalAxonsBuilder().getPaddingHeight(), builderState.getConnectionWeights(), builderState.getBiases());
			
			if (builderState.getConvolutionalAxonsBuilder().getDirectedComponentsContext() != null && 
					builderState.getConvolutionalAxonsBuilder().getAxonsContextConfigurer() != null) {
				AxonsContext axonsContext = axonsComponent.getContext(builderState.getConvolutionalAxonsBuilder().getDirectedComponentsContext(), 0);
				builderState.getConvolutionalAxonsBuilder().getAxonsContextConfigurer().accept(axonsContext);
			}
			
			components.add(axonsComponent);
			
			builderState.setConvolutionalAxonsBuilder(null);
			builderState.getComponentsGraphNeurons().setCurrentNeurons(builderState.getComponentsGraphNeurons().getRightNeurons());
			builderState.getComponentsGraphNeurons().setRightNeurons(null);
			builderState.setConnectionWeights(null);
		} 
		if ((builderState.getMaxPoolingAxonsBuilder() != null) && builderState.getComponentsGraphNeurons().getRightNeurons() != null) {			
			DirectedAxonsComponent<Neurons3D, Neurons3D> axonsComponent
			  = directedAxonsComponentFactory.createMaxPoolingAxonsComponent(builderState.getComponentsGraphNeurons().getCurrentNeurons(), builderState.getComponentsGraphNeurons().getRightNeurons(), builderState.getMaxPoolingAxonsBuilder().getStrideWidth(), builderState.getMaxPoolingAxonsBuilder().getStrideHeight(), builderState.getMaxPoolingAxonsBuilder().getPaddingWidth(), builderState.getMaxPoolingAxonsBuilder().getPaddingHeight(), builderState.getMaxPoolingAxonsBuilder().isScaleOutputs());
			this.components.add(axonsComponent);
			builderState.setMaxPoolingAxonsBuilder(null);
			builderState.getComponentsGraphNeurons().setCurrentNeurons(builderState.getComponentsGraphNeurons().getRightNeurons());
			builderState.getComponentsGraphNeurons().setRightNeurons(null);
			builderState.setConnectionWeights(null);
		}
		if ((builderState.getBatchNormAxonsBuilder() != null) && builderState.getComponentsGraphNeurons().getRightNeurons() != null) {	
			Neurons3D leftNeurons = builderState.getComponentsGraphNeurons().getCurrentNeurons();
			if (builderState.getComponentsGraphNeurons().hasBiasUnit() && !leftNeurons.hasBiasUnit()) {
				leftNeurons = new Neurons3D(builderState.getComponentsGraphNeurons().getCurrentNeurons().getWidth(), builderState.getComponentsGraphNeurons().getCurrentNeurons().getHeight(), builderState.getComponentsGraphNeurons().getCurrentNeurons().getDepth(), true);
			} 
			
			DirectedAxonsComponent<Neurons3D, Neurons3D> axonsComponent
			  = directedAxonsComponentFactory.createBatchNormAxonsComponent(leftNeurons, builderState.getComponentsGraphNeurons().getRightNeurons(), 
					  builderState.getBatchNormAxonsBuilder().getGamma(), builderState.getBatchNormAxonsBuilder().getBeta(),
					  builderState.getBatchNormAxonsBuilder().getMean(), builderState.getBatchNormAxonsBuilder().getVariance());
			
			if (builderState.getBatchNormAxonsBuilder().getDirectedComponentsContext() != null && 
					builderState.getBatchNormAxonsBuilder().getAxonsContextConfigurer() != null) {
				AxonsContext axonsContext = axonsComponent.getContext(builderState.getBatchNormAxonsBuilder().getDirectedComponentsContext(), 0);
				builderState.getBatchNormAxonsBuilder().getAxonsContextConfigurer().accept(axonsContext);
			}
			
			this.components.add(axonsComponent);
			builderState.setBatchNormAxonsBuilder(null);
			builderState.getComponentsGraphNeurons().setCurrentNeurons(builderState.getComponentsGraphNeurons().getRightNeurons());
			builderState.getComponentsGraphNeurons().setRightNeurons(null);
			builderState.setConnectionWeights(null);
		}
		if ((builderState.getAveragePoolingAxonsBuilder() != null) && builderState.getComponentsGraphNeurons().getRightNeurons() != null) {			
			DirectedAxonsComponent<Neurons3D, Neurons3D> axonsComponent
			  = directedAxonsComponentFactory.createAveragePoolingAxonsComponent(builderState.getComponentsGraphNeurons().getCurrentNeurons(), builderState.getComponentsGraphNeurons().getRightNeurons(), builderState.getAveragePoolingAxonsBuilder().getStrideWidth(), builderState.getAveragePoolingAxonsBuilder().getStrideHeight(),builderState.getAveragePoolingAxonsBuilder().getPaddingWidth(), builderState.getAveragePoolingAxonsBuilder().getPaddingHeight());
			this.components.add(axonsComponent);
			builderState.setAveragePoolingAxonsBuilder(null);
			builderState.getComponentsGraphNeurons().setCurrentNeurons(builderState.getComponentsGraphNeurons().getRightNeurons());
			builderState.getComponentsGraphNeurons().setRightNeurons(null);
			builderState.setConnectionWeights(null);
		}
	}

	public Axons3DBuilder withBiasUnit() {
		builderState.getComponentsGraphNeurons().setHasBiasUnit(true);
		return this;
	}
	@Override
	public UncompletedFullyConnectedAxonsBuilder<D> withFullyConnectedAxons() {
		builderState.setConnectionWeights(null);
		builderState.getComponentsGraphNeurons().setHasBiasUnit(false);
		UncompletedFullyConnectedAxonsBuilder<D> axonsBuilder = new UncompletedFullyConnectedAxonsBuilderImpl<>(this::getBuilder, builderState.getComponentsGraphNeurons().getCurrentNeurons());
		builderState.setFullyConnectedAxonsBuilder(axonsBuilder);
		return axonsBuilder;
	}
	
	@Override
	public UncompletedPoolingAxonsBuilder<C> withMaxPoolingAxons() {
		addAxonsIfApplicable();
		UncompletedPoolingAxonsBuilder<C> axonsBuilder = new UncompletedMaxPoolingAxonsBuilderImpl<>(this::get3DBuilder, builderState.getComponentsGraphNeurons().getCurrentNeurons());
		builderState.setMaxPoolingAxonsBuilder(axonsBuilder);
		builderState.setConnectionWeights(null);
		builderState.getComponentsGraphNeurons().setHasBiasUnit(false);
		return axonsBuilder;
	}
	
	@Override
	public UncompletedPoolingAxonsBuilder<C> withAveragePoolingAxons() {
		addAxonsIfApplicable();
		UncompletedPoolingAxonsBuilder<C> axonsBuilder = new UncompletedMaxPoolingAxonsBuilderImpl<>(this::get3DBuilder, builderState.getComponentsGraphNeurons().getCurrentNeurons());
		builderState.setAveragePoolingAxonsBuilder(axonsBuilder);
		builderState.setConnectionWeights(null);
		builderState.getComponentsGraphNeurons().setHasBiasUnit(false);
		return axonsBuilder;
	}
	
	@Override
	public UncompletedBatchNormAxonsBuilder<C> withBatchNormAxons() {
		addAxonsIfApplicable();
		UncompletedBatchNormAxonsBuilder<C> axonsBuilder = new UncompletedBatchNormAxonsBuilderImpl<>(this::get3DBuilder, builderState.getComponentsGraphNeurons().getCurrentNeurons());
		builderState.setBatchNormAxonsBuilder(axonsBuilder);
		builderState.setConnectionWeights(null);
		builderState.getComponentsGraphNeurons().setHasBiasUnit(false);
		return axonsBuilder;
	}
	
	@Override
	public UncompletedConvolutionalAxonsBuilder<C> withConvolutionalAxons() {
		addAxonsIfApplicable();
		UncompletedConvolutionalAxonsBuilder<C> axonsBuilder = new UncompletedConvolutionalAxonsBuilderImpl<>(this::get3DBuilder, builderState.getComponentsGraphNeurons().getCurrentNeurons());
		builderState.setConvolutionalAxonsBuilder(axonsBuilder);
		builderState.setConnectionWeights(null);
		builderState.getComponentsGraphNeurons().setHasBiasUnit(false);
		return axonsBuilder;
	}

	@Override
	public SynapsesAxons3DGraphBuilder<C, D> withSynapses() {
		addAxonsIfApplicable();
		SynapsesAxons3DGraphBuilder<C, D> synapsesBuilder = new SynapsesAxons3DGraphBuilderImpl<>(this::get3DBuilder, this::getBuilder, directedAxonsComponentFactory, builderState, new ArrayList<>());
		builderState.setSynapsesBuilder(synapsesBuilder);
		return synapsesBuilder;
	}

	public void addActivationFunction(DifferentiableActivationFunction activationFunction) {
		addAxonsIfApplicable();
		components.add(new DifferentiableActivationFunctionDirectedComponentImpl(activationFunction));
	}

	public DirectedComponentChain<NeuronsActivation, ?, ?, ?> getComponentChain() {
		addAxonsIfApplicable();
		return new DefaultDirectedComponentChainImpl<>(components);
	}
	
	@Override
	public void addComponents(
			List<ChainableDirectedComponent<NeuronsActivation, ? extends ChainableDirectedComponentActivation<NeuronsActivation>, ?>> components) {
		this.components.addAll(components);
	}

	@Override
	public void addComponent(
			ChainableDirectedComponent<NeuronsActivation, ? extends ChainableDirectedComponentActivation<NeuronsActivation>, ?> component) {
		this.components.add(component);		
	}
}
