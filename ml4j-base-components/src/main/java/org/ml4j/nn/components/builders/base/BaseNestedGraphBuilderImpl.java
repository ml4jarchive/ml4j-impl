package org.ml4j.nn.components.builders.base;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.function.Supplier;

import org.ml4j.nn.components.DefaultChainableDirectedComponent;
import org.ml4j.nn.components.DefaultDirectedComponentChain;
import org.ml4j.nn.components.DefaultDirectedComponentChainActivation;
import org.ml4j.nn.components.PathCombinationStrategy;
import org.ml4j.nn.components.axons.DirectedAxonsComponent;
import org.ml4j.nn.components.builders.BaseGraphBuilderState;
import org.ml4j.nn.components.builders.axons.AxonsBuilder;
import org.ml4j.nn.components.builders.common.ComponentsContainer;
import org.ml4j.nn.components.builders.common.PathEnder;
import org.ml4j.nn.components.defaults.DefaultDirectedComponentChainBatch;
import org.ml4j.nn.components.defaults.DefaultDirectedComponentChainBatchImpl;
import org.ml4j.nn.components.defaults.DefaultDirectedComponentChainBipoleGraphImpl;
import org.ml4j.nn.components.defaults.DefaultDirectedComponentChainImpl;
import org.ml4j.nn.components.factories.DirectedComponentFactory;
import org.ml4j.nn.neurons.Neurons;

public abstract class BaseNestedGraphBuilderImpl<P extends ComponentsContainer<Neurons>, C extends AxonsBuilder> extends BaseGraphBuilderImpl<C> implements PathEnder<P, C>{

	protected Supplier<P> parentGraph;
	private boolean pathEnded;
	private boolean pathsEnded;
	
	public BaseNestedGraphBuilderImpl(Supplier<P> parentGraph, DirectedComponentFactory directedComponentFactory, BaseGraphBuilderState builderState,
			List<DefaultChainableDirectedComponent<?, ?>> components) {
		super(directedComponentFactory, builderState, components);
		this.parentGraph = parentGraph;
	}
	
	protected abstract C createNewNestedGraphBuilder();
	
	protected void completeNestedGraph(boolean addSkipConnection) {

		if (!pathEnded) {
			Neurons initialNeurons = getComponentsGraphNeurons().getCurrentNeurons();
			addAxonsIfApplicable();
			Neurons endNeurons = getComponentsGraphNeurons().getCurrentNeurons();
			DefaultDirectedComponentChain
			chain = new DefaultDirectedComponentChainImpl(getComponents());
			parentGraph.get().getChains().add(chain);
			parentGraph.get().getComponentsGraphNeurons().setCurrentNeurons(getComponentsGraphNeurons().getCurrentNeurons());
			parentGraph.get().getComponentsGraphNeurons().setRightNeurons(getComponentsGraphNeurons().getRightNeurons());
			if (addSkipConnection) {
				if (initialNeurons.getNeuronCountIncludingBias() == endNeurons.getNeuronCountIncludingBias()) {
					
					DirectedAxonsComponent<Neurons, Neurons> skipConnectionAxons = 
							directedComponentFactory.createPassThroughAxonsComponent(initialNeurons, endNeurons);
					DefaultDirectedComponentChain
					skipConnection = new DefaultDirectedComponentChainImpl(Arrays.asList(skipConnectionAxons));
					this.parentGraph.get().getChains().add(skipConnection);
				} else {
					DirectedAxonsComponent<Neurons, Neurons> skipConnectionAxons = directedComponentFactory.createFullyConnectedAxonsComponent(new Neurons(initialNeurons.getNeuronCountExcludingBias(), 
							true), endNeurons, null, null);
					DefaultDirectedComponentChain
					skipConnection = new DefaultDirectedComponentChainImpl(Arrays.asList(skipConnectionAxons));
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
			List<DefaultDirectedComponentChain> chainsList = new ArrayList<>();
			chainsList.addAll(this.parentGraph.get().getChains());
			DefaultDirectedComponentChainBatch<DefaultDirectedComponentChain, DefaultDirectedComponentChainActivation> batch = new DefaultDirectedComponentChainBatchImpl<>(chainsList);
			parentGraph.get().addComponent(new DefaultDirectedComponentChainBipoleGraphImpl<>(directedComponentFactory, batch, pathCombinationStrategy));
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
