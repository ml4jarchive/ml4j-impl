package org.ml4j.nn.components.builders.base;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.function.Supplier;

import org.ml4j.nn.axons.PassThroughAxonsImpl;
import org.ml4j.nn.components.ChainableDirectedComponent;
import org.ml4j.nn.components.ChainableDirectedComponentActivation;
import org.ml4j.nn.components.PathCombinationStrategy;
import org.ml4j.nn.components.axons.DirectedAxonsComponent;
import org.ml4j.nn.components.axons.DirectedAxonsComponentFactory;
import org.ml4j.nn.components.axons.DirectedAxonsComponentImpl;
import org.ml4j.nn.components.builders.BaseGraphBuilderState;
import org.ml4j.nn.components.builders.axons.AxonsBuilder;
import org.ml4j.nn.components.builders.common.ComponentsContainer;
import org.ml4j.nn.components.builders.common.PathEnder;
import org.ml4j.nn.components.defaults.DefaultDirectedComponentChain;
import org.ml4j.nn.components.defaults.DefaultDirectedComponentChainBatch;
import org.ml4j.nn.components.defaults.DefaultDirectedComponentChainBatchImpl;
import org.ml4j.nn.components.defaults.DefaultDirectedComponentChainBipoleGraphImpl;
import org.ml4j.nn.components.defaults.DefaultDirectedComponentChainImpl;
import org.ml4j.nn.neurons.Neurons;
import org.ml4j.nn.neurons.NeuronsActivation;

public abstract class BaseNestedGraphBuilderImpl<P extends ComponentsContainer<Neurons>, C extends AxonsBuilder> extends BaseGraphBuilderImpl<C> implements PathEnder<P, C>{

	protected Supplier<P> parentGraph;
	private boolean pathEnded;
	private boolean pathsEnded;
	private List<DefaultDirectedComponentChain<ChainableDirectedComponentActivation<NeuronsActivation>>> chains;

	
	public BaseNestedGraphBuilderImpl(Supplier<P> parentGraph, DirectedAxonsComponentFactory directedAxonsComponentFactory, BaseGraphBuilderState builderState,
			List<ChainableDirectedComponent<NeuronsActivation, ? extends ChainableDirectedComponentActivation<NeuronsActivation>, ?>> components) {
		super(directedAxonsComponentFactory, builderState, components);
		this.parentGraph = parentGraph;
		this.chains = new ArrayList<>();
	}
	
	protected abstract C createNewNestedGraphBuilder();
	
	protected void completeNestedGraph(boolean addSkipConnection) {

		if (!pathEnded) {
			Neurons initialNeurons = getComponentsGraphNeurons().getCurrentNeurons();
			addAxonsIfApplicable();
			Neurons endNeurons = getComponentsGraphNeurons().getCurrentNeurons();
			DefaultDirectedComponentChain<ChainableDirectedComponentActivation<NeuronsActivation>>
			chain = new DefaultDirectedComponentChainImpl<>(getComponents());
			parentGraph.get().getChains().add(chain);
			parentGraph.get().getComponentsGraphNeurons().setCurrentNeurons(getComponentsGraphNeurons().getCurrentNeurons());
			parentGraph.get().getComponentsGraphNeurons().setRightNeurons(getComponentsGraphNeurons().getRightNeurons());
			if (addSkipConnection) {
				if (initialNeurons.getNeuronCountIncludingBias() == endNeurons.getNeuronCountIncludingBias()) {
					DirectedAxonsComponent<Neurons, Neurons> skipConnectionAxons = new DirectedAxonsComponentImpl<>(
							new PassThroughAxonsImpl(initialNeurons, endNeurons));
					DefaultDirectedComponentChain<ChainableDirectedComponentActivation<NeuronsActivation>>
					skipConnection = new DefaultDirectedComponentChainImpl<>(Arrays.asList(skipConnectionAxons));
					this.parentGraph.get().getChains().add(skipConnection);
				} else {
					DirectedAxonsComponent<Neurons, Neurons> skipConnectionAxons = directedAxonsComponentFactory.createFullyConnectedAxonsComponent(new Neurons(initialNeurons.getNeuronCountExcludingBias(), 
							true), endNeurons, null, null);
					DefaultDirectedComponentChain<ChainableDirectedComponentActivation<NeuronsActivation>>
					skipConnection = new DefaultDirectedComponentChainImpl<>(Arrays.asList(skipConnectionAxons));
					this.parentGraph.get().getChains().add(skipConnection);
				}
			}
			pathEnded = true;
		}
	}
	
	protected void completeNestedGraphs(PathCombinationStrategy pathCombinationStrategy) {
		if (!pathsEnded) {
			parentGraph.get().getComponentsGraphNeurons().setCurrentNeurons(getComponentsGraphNeurons().getCurrentNeurons());
			parentGraph.get().getComponentsGraphNeurons().setRightNeurons(getComponentsGraphNeurons().getRightNeurons());
			parentGraph.get().getComponentsGraphNeurons().setHasBiasUnit(getComponentsGraphNeurons().hasBiasUnit());
			List<DefaultDirectedComponentChain<ChainableDirectedComponentActivation<NeuronsActivation>>> chainsList = new ArrayList<>();
			chainsList.addAll(this.parentGraph.get().getChains());
			DefaultDirectedComponentChainBatch<?, ?> batch = new DefaultDirectedComponentChainBatchImpl<>(chainsList);
			parentGraph.get().addComponent(new DefaultDirectedComponentChainBipoleGraphImpl<>(batch, pathCombinationStrategy));
			pathsEnded = true;
			parentGraph.get().getEndNeurons().clear();
			parentGraph.get().getChains().clear();
		}
	}

	@Override
	public P endParallelPaths(PathCombinationStrategy pathCombinationStrategy) {
		completeNestedGraph(false);
		completeNestedGraphs(pathCombinationStrategy);
		return parentGraph.get();
	}

	@Override
	public C withPath() {
		completeNestedGraph(false);
		return createNewNestedGraphBuilder();
	}
}
