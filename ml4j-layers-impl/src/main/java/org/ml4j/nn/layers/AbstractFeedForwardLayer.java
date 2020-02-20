/*
 * Copyright 2017 the original author or authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.ml4j.nn.layers;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Set;
import java.util.stream.Collectors;

import org.ml4j.nn.axons.Axons;
import org.ml4j.nn.axons.AxonsContext;
import org.ml4j.nn.components.DirectedComponentsContext;
import org.ml4j.nn.components.NeuralComponentBaseType;
import org.ml4j.nn.components.NeuralComponentType;
import org.ml4j.nn.components.NeuralComponentVisitor;
import org.ml4j.nn.components.factories.DirectedComponentFactory;
import org.ml4j.nn.components.onetone.DefaultChainableDirectedComponent;
import org.ml4j.nn.components.onetone.DefaultDirectedComponentChain;
import org.ml4j.nn.components.onetone.TrailingActivationFunctionDirectedComponentChain;
import org.ml4j.nn.components.onetone.TrailingActivationFunctionDirectedComponentChainActivation;
import org.ml4j.nn.components.onetoone.TrailingActivationFunctionDirectedComponentChainImpl;
import org.ml4j.nn.neurons.NeuronsActivation;
import org.ml4j.nn.neurons.NeuronsActivationFeatureOrientation;
import org.ml4j.nn.neurons.format.NeuronsActivationFormat;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * A default base implementation of FeedForwardLayer.
 * 
 * @author Michael Lavelle
 * 
 * @param <A> The type of primary Axons in this FeedForwardLayer.
 */
public abstract class AbstractFeedForwardLayer<A extends Axons<?, ?, ?>, L extends FeedForwardLayer<A, L>>
		implements FeedForwardLayer<A, L> {
	/**
	 * Default serialization id.
	 */
	private static final long serialVersionUID = 1L;

	private static final Logger LOGGER = LoggerFactory.getLogger(AbstractFeedForwardLayer.class);

	protected String name;

	protected TrailingActivationFunctionDirectedComponentChain trailingActivationFunctionDirectedComponentChain;
	
	/**
	 * @param name The name of the layer.
	 * @param trailingActivationFunctionDirectedComponentChain The chain of components, ending with an activation function.
	 */
	protected AbstractFeedForwardLayer(String name,
			TrailingActivationFunctionDirectedComponentChain trailingActivationFunctionDirectedComponentChain) {
		this.name = name;
		this.trailingActivationFunctionDirectedComponentChain = trailingActivationFunctionDirectedComponentChain;
	}

	/**
	 * 
	 * @param name The name of the layer.
	 * @param directedComponentFactory The directed component factory.
	 * @param componentChain The initialising component chain.
	 */
	protected AbstractFeedForwardLayer(String name, DirectedComponentFactory directedComponentFactory,
			DefaultDirectedComponentChain componentChain) {
		this.trailingActivationFunctionDirectedComponentChain = new TrailingActivationFunctionDirectedComponentChainImpl(
				directedComponentFactory, componentChain.decompose());
		this.name = name;
	}
	
	

	@Override
	public DirectedLayerActivation forwardPropagate(NeuronsActivation inputNeuronsActivation,
			DirectedLayerContext directedLayerContext) {
		LOGGER.debug(directedLayerContext.toString() + ":Forward propagating through layer");

		TrailingActivationFunctionDirectedComponentChainActivation activation = trailingActivationFunctionDirectedComponentChain
				.forwardPropagate(inputNeuronsActivation, directedLayerContext.getDirectedComponentsContext());

		return new DirectedLayerActivationImpl(this, activation, directedLayerContext);
	}

	@Override
	public DirectedLayerContext getContext(DirectedComponentsContext directedComponentsContext) {
		return directedComponentsContext.getContext(this, () -> new DirectedLayerContextImpl(this, 
				directedComponentsContext), context -> { DirectedLayerContext layerContext = new DirectedLayerContextImpl(this, 
						directedComponentsContext); layerContext.withFreezeOut(context.isWithFreezeOut()); return layerContext; });
	}

	@Override
	public List<DefaultChainableDirectedComponent<?, ?>> decompose() {
		return getComponents().stream().flatMap(c -> c.decompose().stream()).collect((Collectors.toList()));
	}
	

	@Override
	public Set<DefaultChainableDirectedComponent<?, ?>> flatten() {
		Set<DefaultChainableDirectedComponent<?, ?>> allComponentsIncludingThis = new HashSet<>(Arrays.asList(this));
		allComponentsIncludingThis.addAll(trailingActivationFunctionDirectedComponentChain.flatten());
		return allComponentsIncludingThis;
	}

	

	@Override
	public NeuralComponentType getComponentType() {
		return NeuralComponentType.createSubType(NeuralComponentType.getBaseType(NeuralComponentBaseType.LAYER),
				FeedForwardLayer.class.getName());
	}
	
	@Override
	public boolean isSupported(NeuronsActivationFormat<?> format) {
		return NeuronsActivationFeatureOrientation.ROWS_SPAN_FEATURE_SET.equals(format.getFeatureOrientation());
	}

	@Override
	public Optional<NeuronsActivationFormat<?>> optimisedFor() {
		return Optional.empty();
	}

	@Override
	public String getName() {
		return name;
	}

	@Override
	public List<DefaultChainableDirectedComponent<?, ?>> getComponents() {
		List<DefaultChainableDirectedComponent<?, ?>> components = new ArrayList<>();
		components.addAll(trailingActivationFunctionDirectedComponentChain.getComponents());
		return components;
	}

	@Override
	public String accept(NeuralComponentVisitor<DefaultChainableDirectedComponent<?, ?>> visitor) {
		return visitor.visitSequentialComponentChain(getComponents());
	}
	
	protected abstract String getPrimaryAxonsComponentName();
	
	protected AxonsContext getAxonsContext(DirectedComponentsContext directedComponentsContext, String axonsComponentName) {
		System.out.println("Getting axons context:" + directedComponentsContext.isTrainingContext());
		Map<String, AxonsContext> nestedAxonsContextsByComponentName = getNestedContexts(directedComponentsContext, AxonsContext.class);
		AxonsContext primaryAxonsContext = nestedAxonsContextsByComponentName.get(axonsComponentName);
		if (primaryAxonsContext == null) {
			throw new IllegalStateException("Unable to lookup primary axons components context for component name:" + axonsComponentName);
		} else {
			return primaryAxonsContext;
		}
	}

}
