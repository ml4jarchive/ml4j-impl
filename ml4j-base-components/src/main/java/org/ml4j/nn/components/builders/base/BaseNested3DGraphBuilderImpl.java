/*
 * Copyright 2019 the original author or authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
 * in compliance with the License. You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software distributed under the License
 * is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
 * or implied. See the License for the specific language governing permissions and limitations under
 * the License.
 */
package org.ml4j.nn.components.builders.base;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.function.Supplier;

import org.ml4j.nn.components.axons.DirectedAxonsComponent;
import org.ml4j.nn.components.builders.Base3DGraphBuilderState;
import org.ml4j.nn.components.builders.axons.Axons3DBuilder;
import org.ml4j.nn.components.builders.axons.AxonsBuilder;
import org.ml4j.nn.components.builders.common.ComponentsContainer;
import org.ml4j.nn.components.factories.DirectedComponentFactory;
import org.ml4j.nn.components.manytomany.DefaultDirectedComponentChainBatch;
import org.ml4j.nn.components.manytoone.PathCombinationStrategy;
import org.ml4j.nn.components.onetone.DefaultChainableDirectedComponent;
import org.ml4j.nn.components.onetone.DefaultDirectedComponentChain;
import org.ml4j.nn.neurons.Neurons;
import org.ml4j.nn.neurons.Neurons3D;

public abstract class BaseNested3DGraphBuilderImpl<P extends ComponentsContainer<Neurons3D>, 
		C extends Axons3DBuilder, D extends AxonsBuilder> extends Base3DGraphBuilderImpl<C, D> {
	
	protected Supplier<P> parent3DGraph;
	private boolean pathEnded;
	private boolean pathsEnded;
	
	public BaseNested3DGraphBuilderImpl(Supplier<P> parent3DGraph, DirectedComponentFactory directedComponentFactory,
			Base3DGraphBuilderState builderState,
			List<DefaultChainableDirectedComponent<?, ?>> components) {
		super(directedComponentFactory, builderState, components);
		this.parent3DGraph = parent3DGraph;
	}
	
	protected abstract C createNewNestedGraphBuilder();
	
	protected void completeNestedGraph(boolean addSkipConnection) {
		if (!pathEnded) {	
			Neurons3D initialNeurons = getComponentsGraphNeurons().getCurrentNeurons();
			addAxonsIfApplicable();
			Neurons3D endNeurons = getComponentsGraphNeurons().getCurrentNeurons();
			DefaultDirectedComponentChain
			chain = directedComponentFactory.createDirectedComponentChain(getComponents());
			this.parent3DGraph.get().getChains().add(chain);
			this.parent3DGraph.get().getEndNeurons().add(getComponentsGraphNeurons().getCurrentNeurons());
			if (addSkipConnection) {
				if (initialNeurons.getNeuronCountIncludingBias() == endNeurons.getNeuronCountIncludingBias()) {
					
					DirectedAxonsComponent<Neurons, Neurons, ?> skipConnectionAxons =
							directedComponentFactory.createPassThroughAxonsComponent(initialNeurons, endNeurons);
					DefaultDirectedComponentChain
					skipConnection = directedComponentFactory.createDirectedComponentChain(Arrays.asList(skipConnectionAxons));
					this.parent3DGraph.get().getChains().add(skipConnection);
				} else {
					
					DirectedAxonsComponent<Neurons, Neurons, ?> skipConnectionAxons = 
							directedComponentFactory.createFullyConnectedAxonsComponent(new Neurons(initialNeurons.getNeuronCountExcludingBias(), true), endNeurons, null, null);
					DefaultDirectedComponentChain
					skipConnection = directedComponentFactory.createDirectedComponentChain(Arrays.asList(skipConnectionAxons));
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
			
			List<DefaultDirectedComponentChain> chainsList = new ArrayList<>();
			chainsList.addAll(this.parent3DGraph.get().getChains());
			Neurons graphInputNeurons = chainsList.get(0).getInputNeurons();
			DefaultDirectedComponentChainBatch batch = directedComponentFactory.createDirectedComponentChainBatch(chainsList);
			parent3DGraph.get().addComponent(directedComponentFactory.createDirectedComponentBipoleGraph(graphInputNeurons, parent3DGraph.get().getComponentsGraphNeurons().getCurrentNeurons(), batch, pathCombinationStrategy));
					
					
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
