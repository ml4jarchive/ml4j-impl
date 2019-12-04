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
import org.ml4j.nn.components.builders.Base3DGraphBuilderState;
import org.ml4j.nn.components.builders.axons.Axons3DBuilder;
import org.ml4j.nn.components.builders.axons.AxonsBuilder;
import org.ml4j.nn.components.builders.common.ComponentsContainer;
import org.ml4j.nn.components.defaults.DefaultDirectedComponentChain;
import org.ml4j.nn.components.defaults.DefaultDirectedComponentChainBatch;
import org.ml4j.nn.components.defaults.DefaultDirectedComponentChainBatchImpl;
import org.ml4j.nn.components.defaults.DefaultDirectedComponentChainBipoleGraphImpl;
import org.ml4j.nn.components.defaults.DefaultDirectedComponentChainImpl;
import org.ml4j.nn.neurons.Neurons;
import org.ml4j.nn.neurons.Neurons3D;
import org.ml4j.nn.neurons.NeuronsActivation;

public abstract class BaseNested3DGraphBuilderImpl<P extends ComponentsContainer<Neurons3D>, 
		C extends Axons3DBuilder, D extends AxonsBuilder> extends Base3DGraphBuilderImpl<C, D> {
	
	protected Supplier<P> parent3DGraph;
	private boolean pathEnded;
	private boolean pathsEnded;
	
	public BaseNested3DGraphBuilderImpl(Supplier<P> parent3DGraph, DirectedAxonsComponentFactory directedAxonsComponentFactory,
			Base3DGraphBuilderState builderState,
			List<ChainableDirectedComponent<NeuronsActivation, ? extends ChainableDirectedComponentActivation<NeuronsActivation>, ?>> components) {
		super(directedAxonsComponentFactory, builderState, components);
		this.parent3DGraph = parent3DGraph;
	}
	
	protected abstract C createNewNestedGraphBuilder();
	
	protected void completeNestedGraph(boolean addSkipConnection) {
		if (!pathEnded) {	
			Neurons3D initialNeurons = getComponentsGraphNeurons().getCurrentNeurons();
			addAxonsIfApplicable();
			Neurons3D endNeurons = getComponentsGraphNeurons().getCurrentNeurons();
			DefaultDirectedComponentChain<ChainableDirectedComponentActivation<NeuronsActivation>>
			chain = new DefaultDirectedComponentChainImpl<>(getComponents());
			this.parent3DGraph.get().getChains().add(chain);
			this.parent3DGraph.get().getEndNeurons().add(getComponentsGraphNeurons().getCurrentNeurons());
			if (addSkipConnection) {
				if (initialNeurons.getNeuronCountIncludingBias() == endNeurons.getNeuronCountIncludingBias()) {
					DirectedAxonsComponent<Neurons, Neurons> skipConnectionAxons = new DirectedAxonsComponentImpl<>(
							new PassThroughAxonsImpl(initialNeurons, endNeurons));
					DefaultDirectedComponentChain<ChainableDirectedComponentActivation<NeuronsActivation>>
					skipConnection = new DefaultDirectedComponentChainImpl<>(Arrays.asList(skipConnectionAxons));
					this.parent3DGraph.get().getChains().add(skipConnection);
				} else {
					
					DirectedAxonsComponent<Neurons, Neurons> skipConnectionAxons = 
							directedAxonsComponentFactory.createFullyConnectedAxonsComponent(new Neurons(initialNeurons.getNeuronCountExcludingBias(), true), endNeurons, null, null);
					DefaultDirectedComponentChain<ChainableDirectedComponentActivation<NeuronsActivation>>
					skipConnection = new DefaultDirectedComponentChainImpl<>(Arrays.asList(skipConnectionAxons));
					this.parent3DGraph.get().getChains().add(skipConnection);
				}
			}
			pathEnded = true;
		}
	}
	
	protected void completeNestedGraphs(PathCombinationStrategy pathCombinationStrategy) {
		if (!pathsEnded) {
			if (pathCombinationStrategy ==  PathCombinationStrategy.FILTER_CONCAT) {
				Neurons3D previousNeurons = null;
				int totalDepth = 0;
				for (Neurons3D endNeuronsInstance : this.parent3DGraph.get().getEndNeurons()) {
					if (previousNeurons != null)
					{
						if (previousNeurons.getWidth() != endNeuronsInstance.getWidth()) {
							throw new IllegalStateException("Width doesn't match");
						}
						if (previousNeurons.getHeight() != endNeuronsInstance.getHeight()) {
							throw new IllegalStateException("Height doesn't match");
						}
					}
					totalDepth = totalDepth + endNeuronsInstance.getDepth();
					previousNeurons = endNeuronsInstance;
				}
				parent3DGraph.get().getComponentsGraphNeurons().setCurrentNeurons(new Neurons3D(previousNeurons.getWidth(), previousNeurons.getHeight(), totalDepth, previousNeurons.hasBiasUnit()));
				parent3DGraph.get().getComponentsGraphNeurons().setRightNeurons(getComponentsGraphNeurons().getRightNeurons());
				parent3DGraph.get().getComponentsGraphNeurons().setHasBiasUnit(getComponentsGraphNeurons().hasBiasUnit());
			} else {
				parent3DGraph.get().getComponentsGraphNeurons().setCurrentNeurons(getComponentsGraphNeurons().getCurrentNeurons());
				parent3DGraph.get().getComponentsGraphNeurons().setRightNeurons(getComponentsGraphNeurons().getRightNeurons());
				parent3DGraph.get().getComponentsGraphNeurons().setHasBiasUnit(getComponentsGraphNeurons().hasBiasUnit());
			}
			
			List<DefaultDirectedComponentChain<ChainableDirectedComponentActivation<NeuronsActivation>>> chainsList = new ArrayList<>();
			chainsList.addAll(this.parent3DGraph.get().getChains());
			DefaultDirectedComponentChainBatch<?, ?> batch = new DefaultDirectedComponentChainBatchImpl<>(chainsList);
			parent3DGraph.get().addComponent(new DefaultDirectedComponentChainBipoleGraphImpl<>(batch, pathCombinationStrategy));
			parent3DGraph.get().getEndNeurons().clear();
			parent3DGraph.get().getChains().clear();

			pathsEnded = true;
		}
	}

	public P endParallelPaths(PathCombinationStrategy pathCombinationStrategy) {
		completeNestedGraph(false);
		completeNestedGraphs(pathCombinationStrategy);
		return parent3DGraph.get();
	}

}
