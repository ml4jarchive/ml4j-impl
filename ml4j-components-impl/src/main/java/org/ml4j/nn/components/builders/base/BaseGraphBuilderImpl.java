package org.ml4j.nn.components.builders.base;

import java.util.ArrayList;
import java.util.List;

import org.ml4j.Matrix;
import org.ml4j.nn.activationfunctions.DifferentiableActivationFunction;
import org.ml4j.nn.activationfunctions.DifferentiableActivationFunctionDirectedComponentImpl;
import org.ml4j.nn.axons.AxonsContext;
import org.ml4j.nn.components.ChainableDirectedComponent;
import org.ml4j.nn.components.ChainableDirectedComponentActivation;
import org.ml4j.nn.components.DirectedComponentChain;
import org.ml4j.nn.components.axons.DirectedAxonsComponent;
import org.ml4j.nn.components.axons.DirectedAxonsComponentFactory;
import org.ml4j.nn.components.builders.BaseGraphBuilderState;
import org.ml4j.nn.components.builders.axons.AxonsBuilder;
import org.ml4j.nn.components.builders.axons.AxonsPermitted;
import org.ml4j.nn.components.builders.axons.UncompletedFullyConnectedAxonsBuilder;
import org.ml4j.nn.components.builders.axons.UncompletedFullyConnectedAxonsBuilderImpl;
import org.ml4j.nn.components.builders.common.ComponentsContainer;
import org.ml4j.nn.components.builders.componentsgraph.ComponentsGraphNeurons;
import org.ml4j.nn.components.builders.synapses.SynapsesAxonsGraphBuilder;
import org.ml4j.nn.components.builders.synapses.SynapsesAxonsGraphBuilderImpl;
import org.ml4j.nn.components.builders.synapses.SynapsesPermitted;
import org.ml4j.nn.components.defaults.DefaultDirectedComponentChain;
import org.ml4j.nn.components.defaults.DefaultDirectedComponentChainImpl;
import org.ml4j.nn.neurons.Neurons;
import org.ml4j.nn.neurons.NeuronsActivation;

public abstract class BaseGraphBuilderImpl<C extends AxonsBuilder> implements AxonsPermitted<C>, SynapsesPermitted<C>, AxonsBuilder {

	protected DirectedAxonsComponentFactory directedAxonsComponentFactory;
	
	protected BaseGraphBuilderState builderState;
	
	protected BaseGraphBuilderState initialBuilderState;
	
	private List<DefaultDirectedComponentChain<ChainableDirectedComponentActivation<NeuronsActivation>>> chains;
	private List<Neurons> endNeurons;
	
	protected List<ChainableDirectedComponent<NeuronsActivation, ? extends ChainableDirectedComponentActivation<NeuronsActivation>, ?>> components;
	
	public BaseGraphBuilderImpl(DirectedAxonsComponentFactory directedAxonsComponentFactory, BaseGraphBuilderState builderState, List<ChainableDirectedComponent<NeuronsActivation, ? extends ChainableDirectedComponentActivation<NeuronsActivation>, ?>> components) {
		this.builderState = builderState;
		this.components = components;
		this.directedAxonsComponentFactory = directedAxonsComponentFactory;
		// TODO ML
		this.initialBuilderState = new 	BaseGraphBuilderStateImpl(builderState.getComponentsGraphNeurons().getCurrentNeurons());
		this.chains = new ArrayList<>();
		this.endNeurons = new ArrayList<>();
	}
	
	@Override
	public List<ChainableDirectedComponent<NeuronsActivation, ? extends ChainableDirectedComponentActivation<NeuronsActivation>, ?>> getComponents() {
		return components;
	}
	
	@Override
	public ComponentsContainer<Neurons> getAxonsBuilder() {
		return this;
	}
	
	public BaseGraphBuilderState getBuilderState() {
		return builderState;
	}

	@Override
	public ComponentsGraphNeurons<Neurons> getComponentsGraphNeurons() {
		return builderState.getComponentsGraphNeurons();
	}

	@Override
	public Matrix getConnectionWeights() {
		return builderState.getConnectionWeights();
	}

	public AxonsBuilder withConnectionWeights(Matrix connectionWeights) {
		builderState.setConnectionWeights(connectionWeights);
		return this;
	}
	
	public abstract C getBuilder();

	public void addAxonsIfApplicable() {
		if ((builderState.getFullyConnectedAxonsBuilder() != null) && builderState.getComponentsGraphNeurons().getRightNeurons() != null) {
			Neurons leftNeurons = builderState.getComponentsGraphNeurons().getCurrentNeurons();
			if (builderState.getComponentsGraphNeurons().hasBiasUnit() && !leftNeurons.hasBiasUnit()) {
				leftNeurons = new Neurons(builderState.getComponentsGraphNeurons().getCurrentNeurons().getNeuronCountExcludingBias(), true);
			}

			DirectedAxonsComponent<Neurons, Neurons> axonsComponent = directedAxonsComponentFactory.createFullyConnectedAxonsComponent(leftNeurons, 
					builderState.getComponentsGraphNeurons().getRightNeurons(), builderState.getConnectionWeights(), builderState.getBiases());
					
			if (builderState.getFullyConnectedAxonsBuilder().getDirectedComponentsContext() != null && 
					builderState.getFullyConnectedAxonsBuilder().getAxonsContextConfigurer() != null) {
				AxonsContext axonsContext = axonsComponent.getContext(builderState.getFullyConnectedAxonsBuilder().getDirectedComponentsContext(), 0);
				builderState.getFullyConnectedAxonsBuilder().getAxonsContextConfigurer().accept(axonsContext);
			}
			
			components.add(axonsComponent);
		
			builderState.setFullyConnectedAxonsBuilder(null);
			builderState.getComponentsGraphNeurons().setCurrentNeurons(builderState.getComponentsGraphNeurons().getRightNeurons());
			builderState.getComponentsGraphNeurons().setRightNeurons(null);
		}	
	}
	
	@Override
	public UncompletedFullyConnectedAxonsBuilder<C> withFullyConnectedAxons() {
		addAxonsIfApplicable();
		builderState.setConnectionWeights(null);
		builderState.getComponentsGraphNeurons().setHasBiasUnit(false);
		UncompletedFullyConnectedAxonsBuilder<C> axonsBuilder = 
				new UncompletedFullyConnectedAxonsBuilderImpl<>(this::getBuilder, builderState.getComponentsGraphNeurons().getCurrentNeurons());
		builderState.setFullyConnectedAxonsBuilder(axonsBuilder);
		builderState.getComponentsGraphNeurons().setHasBiasUnit(false);
		return axonsBuilder;
	}

	@Override
	public SynapsesAxonsGraphBuilder<C> withSynapses() {
		addAxonsIfApplicable();
		SynapsesAxonsGraphBuilder<C> synapsesBuilder = new SynapsesAxonsGraphBuilderImpl<>(this::getBuilder, directedAxonsComponentFactory, builderState, new ArrayList<>());
		builderState.setSynapsesBuilder(synapsesBuilder);
		return synapsesBuilder;
	}
	
	protected void addActivationFunction(DifferentiableActivationFunction activationFunction) {
		addAxonsIfApplicable();
		components.add(new DifferentiableActivationFunctionDirectedComponentImpl(activationFunction));
	}
	
	public DirectedComponentChain<NeuronsActivation, ?, ?, ?> getComponentChain() {
		addAxonsIfApplicable();
		return new DefaultDirectedComponentChainImpl<>(components);
	}

	public AxonsBuilder withBiasUnit() {
		builderState.getComponentsGraphNeurons().setHasBiasUnit(true);
		return this;
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

	public List<DefaultDirectedComponentChain<ChainableDirectedComponentActivation<NeuronsActivation>>> getChains() {
		return chains;
	}

	public List<Neurons> getEndNeurons() {
		return endNeurons;
	}
	
	
}
