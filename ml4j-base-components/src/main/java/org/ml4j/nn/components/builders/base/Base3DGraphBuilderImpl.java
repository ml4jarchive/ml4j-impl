package org.ml4j.nn.components.builders.base;

import java.util.ArrayList;
import java.util.List;

import org.ml4j.InterrimMatrix;
import org.ml4j.Matrix;
import org.ml4j.nn.activationfunctions.DifferentiableActivationFunction;
import org.ml4j.nn.axons.Axons3DConfig;
import org.ml4j.nn.axons.AxonsContext;
import org.ml4j.nn.components.axons.DirectedAxonsComponent;
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
import org.ml4j.nn.components.factories.DirectedComponentFactory;
import org.ml4j.nn.components.onetone.DefaultChainableDirectedComponent;
import org.ml4j.nn.components.onetone.DefaultDirectedComponentChain;
import org.ml4j.nn.neurons.Neurons;
import org.ml4j.nn.neurons.Neurons3D;

public abstract class Base3DGraphBuilderImpl<C extends Axons3DBuilder, D extends AxonsBuilder> implements Axons3DPermitted<C, D>, Synapses3DPermitted<C, D>, Axons3DBuilder {

	protected DirectedComponentFactory directedComponentFactory;
	
	protected Base3DGraphBuilderState initialBuilderState;
	
	private List<DefaultChainableDirectedComponent<?, ?>> components;
	
	private List<DefaultDirectedComponentChain> chains;
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
	public List<DefaultChainableDirectedComponent<?, ?>> getComponents() {
		addAxonsIfApplicable();
		return components;
	}
	
	@Override
	public ComponentsGraphNeurons<Neurons3D> getComponentsGraphNeurons() {
		return builderState.getComponentsGraphNeurons();
	}

	public Base3DGraphBuilderImpl(DirectedComponentFactory directedComponentFactory, Base3DGraphBuilderState builderState, List<DefaultChainableDirectedComponent<?, ?>> components) {
		this.components = components;
		this.directedComponentFactory = directedComponentFactory;
		this.builderState = builderState;
		this.initialBuilderState = new 	Base3DGraphBuilderStateImpl(builderState.getComponentsGraphNeurons().getCurrentNeurons());
		this.endNeurons = new ArrayList<>();
		this.chains = new ArrayList<>();

	}
	
	public List<DefaultDirectedComponentChain> getChains() {
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
			
			Axons3DConfig axons3DConfig = new Axons3DConfig().withStrideWidth(builderState.getConvolutionalAxonsBuilder().getStrideWidth())
					.withStrideHeight(builderState.getConvolutionalAxonsBuilder().getStrideHeight())
					.withPaddingWidth(builderState.getConvolutionalAxonsBuilder().getPaddingWidth()).withPaddingHeight(builderState.getConvolutionalAxonsBuilder().getPaddingHeight());
			
			DirectedAxonsComponent<Neurons3D, Neurons3D, ?> axonsComponent
					 = directedComponentFactory.createConvolutionalAxonsComponent(leftNeurons, builderState.getComponentsGraphNeurons().getRightNeurons(), axons3DConfig, builderState.getConnectionWeights(), builderState.getBiases());
			
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
			Axons3DConfig axons3DConfig = new Axons3DConfig().withStrideWidth(builderState.getMaxPoolingAxonsBuilder().getStrideWidth())
					.withStrideHeight(builderState.getMaxPoolingAxonsBuilder().getStrideHeight())
					.withPaddingWidth(builderState.getMaxPoolingAxonsBuilder().getPaddingWidth()).withPaddingHeight(builderState.getMaxPoolingAxonsBuilder().getPaddingHeight());
			DirectedAxonsComponent<Neurons3D, Neurons3D, ?> axonsComponent
			  = directedComponentFactory.createMaxPoolingAxonsComponent(builderState.getComponentsGraphNeurons().getCurrentNeurons(), builderState.getComponentsGraphNeurons().getRightNeurons(), axons3DConfig, builderState.getMaxPoolingAxonsBuilder().isScaleOutputs());
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
			
			DirectedAxonsComponent<Neurons3D, Neurons3D, ?> axonsComponent
			  = directedComponentFactory.createConvolutionalBatchNormAxonsComponent(leftNeurons, builderState.getComponentsGraphNeurons().getRightNeurons(), 
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
			Axons3DConfig axons3DConfig = new Axons3DConfig().withStrideWidth(builderState.getAveragePoolingAxonsBuilder().getStrideWidth())
					.withStrideHeight(builderState.getAveragePoolingAxonsBuilder().getStrideHeight())
					.withPaddingWidth(builderState.getAveragePoolingAxonsBuilder().getPaddingWidth()).withPaddingHeight(builderState.getAveragePoolingAxonsBuilder().getPaddingHeight());
			DirectedAxonsComponent<Neurons3D, Neurons3D, ?> axonsComponent
			  = directedComponentFactory.createAveragePoolingAxonsComponent(builderState.getComponentsGraphNeurons().getCurrentNeurons(), builderState.getComponentsGraphNeurons().getRightNeurons(), axons3DConfig);
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
		SynapsesAxons3DGraphBuilder<C, D> synapsesBuilder = new SynapsesAxons3DGraphBuilderImpl<>(this::get3DBuilder, this::getBuilder, directedComponentFactory, builderState, new ArrayList<>());
		builderState.setSynapsesBuilder(synapsesBuilder);
		return synapsesBuilder;
	}

	public void addActivationFunction(DifferentiableActivationFunction activationFunction) {
		addAxonsIfApplicable();
		components.add(directedComponentFactory.createDifferentiableActivationFunctionComponent(this.builderState.getComponentsGraphNeurons().getCurrentNeurons(), activationFunction));
	}

	public DefaultDirectedComponentChain getComponentChain() {
		addAxonsIfApplicable();
		return directedComponentFactory.createDirectedComponentChain(components);
	}
	
	@Override
	public void addComponents(
			List<DefaultChainableDirectedComponent<?, ?>> components) {
		this.components.addAll(components);
	}

	@Override
	public void addComponent(
			DefaultChainableDirectedComponent<?, ?> component) {
		this.components.add(component);		
	}
}
