/*
 * Copyright 2020 the original author or authors.
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
package org.ml4j.nn.sessions;

import java.util.ArrayList;

import org.ml4j.nn.components.DirectedComponentsContext;
import org.ml4j.nn.components.factories.DirectedComponentFactory;
import org.ml4j.nn.components.onetone.DefaultChainableDirectedComponent;
import org.ml4j.nn.layers.DirectedLayerFactory;
import org.ml4j.nn.neurons.Neurons;
import org.ml4j.nn.neurons.Neurons3D;
import org.ml4j.nn.supervised.LayeredSupervisedFeedForwardNeuralNetworkFactory;
import org.ml4j.nn.supervised.SupervisedFeedForwardNeuralNetworkFactory;

/**
 * Implementation of SessionImpl for computation graphs formed of DefaultChainableDirectedComponent<?, ?>
 * 
 * @author Michael Lavelle
 */
public class DefaultSessionImpl extends SessionImpl<DefaultChainableDirectedComponent<?, ?>> implements DefaultSession {

	private DirectedComponentFactory directedComponentFactory;
	private DirectedLayerFactory directedLayerFactory;
	private SupervisedFeedForwardNeuralNetworkFactory supervisedFeedForwardNeuralNetworkFactory;
	private LayeredSupervisedFeedForwardNeuralNetworkFactory layeredSupervisedFeedForwardNeuralNetworkFactory;

	public DefaultSessionImpl(DirectedComponentFactory directedComponentFactory,
			DirectedLayerFactory directedLayerFactory,
			 SupervisedFeedForwardNeuralNetworkFactory supervisedFeedForwardNeuralNetworkFactory,
			 LayeredSupervisedFeedForwardNeuralNetworkFactory layeredSupervisedFeedForwardNeuralNetworkFactory,
			DirectedComponentsContext directedComponentsContext) {
		super(directedComponentFactory, directedComponentsContext);
		this.directedComponentFactory = directedComponentFactory;
		this.directedLayerFactory = directedLayerFactory;
		this.supervisedFeedForwardNeuralNetworkFactory = supervisedFeedForwardNeuralNetworkFactory;
		this.layeredSupervisedFeedForwardNeuralNetworkFactory = layeredSupervisedFeedForwardNeuralNetworkFactory;

	}
	
	public DefaultSessionImpl(DirectedComponentFactory directedComponentFactory,
			DirectedLayerFactory directedLayerFactory,
			 SupervisedFeedForwardNeuralNetworkFactory supervisedFeedForwardNeuralNetworkFactory,
			DirectedComponentsContext directedComponentsContext) {
		this(directedComponentFactory, directedLayerFactory, supervisedFeedForwardNeuralNetworkFactory, null, directedComponentsContext);
	}
	
	@Override
	public DirectedComponentFactory getNeuralComponentFactory() {
		return directedComponentFactory;
	}

	@Override
	public DirectedLayerFactory getDirectedLayerFactory() {
		return directedLayerFactory;
	}

	@Override
	public SupervisedFeedForwardNeuralNetworkFactory getSupervisedFeedForwardNeuralNetworkFactory() {
		return supervisedFeedForwardNeuralNetworkFactory;
	}

	@Override
	public SupervisedFeedForwardNeuralNetwork3DBuilderSession buildSupervised3DNeuralNetwork(String networkName, Neurons3D initialNeurons) {
		return new DefaultSupervisedFeedForwardNeuralNetwork3DBuilderSession(directedComponentFactory, directedLayerFactory, supervisedFeedForwardNeuralNetworkFactory, networkName, getDirectedComponentsContext(), new ArrayList<>(), initialNeurons);
	}
	
	@Override
	public SupervisedFeedForwardNeuralNetworkBuilderSession buildSupervisedNeuralNetwork(String networkName, Neurons initialNeurons) {
		return new DefaultSupervisedFeedForwardNeuralNetworkBuilderSession(directedComponentFactory, directedLayerFactory, supervisedFeedForwardNeuralNetworkFactory, networkName, getDirectedComponentsContext(), new ArrayList<>(), initialNeurons);
	}

	@Override
	public DirectedLayerBuilderSession buildLayer() {
		if (directedLayerFactory == null) {
			throw new IllegalStateException("No DirectedLayerFactory has been configured on the Session");
		}
		return new DefaultDirectedLayerBuilderSession(directedLayerFactory);
	}

	@Override
	public LayeredSupervisedFeedForwardNeuralNetworkFactory getLayeredSupervisedFeedForwardNeuralNetworkFactory() {
		return layeredSupervisedFeedForwardNeuralNetworkFactory;
	}

	@Override
	public LayeredSupervisedFeedForwardNeuralNetwork3DBuilderSession buildLayeredSupervised3DNeuralNetwork(String networkName) {
		return new DefaultLayeredSupervisedFeedForwardNeuralNetwork3DBuilderSession(directedComponentFactory, directedLayerFactory, layeredSupervisedFeedForwardNeuralNetworkFactory, new ArrayList<>(), networkName);
	}

	@Override
	public LayeredSupervisedFeedForwardNeuralNetworkBuilderSession buildLayeredSupervisedNeuralNetwork(String networkName) {
		return new DefaultLayeredSupervisedFeedForwardNeuralNetworkBuilderSession(directedComponentFactory, directedLayerFactory, layeredSupervisedFeedForwardNeuralNetworkFactory, new ArrayList<>(), networkName);
	}
}
