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
package org.ml4j.nn.sessions.factories;

import org.ml4j.MatrixFactory;
import org.ml4j.nn.components.DirectedComponentsContext;
import org.ml4j.nn.components.DirectedComponentsContextImpl;
import org.ml4j.nn.components.factories.DirectedComponentFactory;
import org.ml4j.nn.layers.DirectedLayerFactory;
import org.ml4j.nn.sessions.DefaultSession;
import org.ml4j.nn.sessions.DefaultSessionImpl;
import org.ml4j.nn.supervised.LayeredSupervisedFeedForwardNeuralNetworkFactory;
import org.ml4j.nn.supervised.SupervisedFeedForwardNeuralNetworkFactory;

/**
 * Factory implementation for Sessions with DefaultChainableDirectedComponent components
 * 
 * @author Michael Lavelle
 *
 */
public class DefaultSessionFactoryImpl implements DefaultSessionFactory{

	protected DirectedComponentFactory directedComponentFactory;
	protected MatrixFactory matrixFactory;
	protected DirectedLayerFactory directedLayerFactory;
	protected SupervisedFeedForwardNeuralNetworkFactory supervisedFeedForwardNeuralNetworkFactory;
	protected LayeredSupervisedFeedForwardNeuralNetworkFactory layeredSupervisedFeedForwardNeuralNetworkFactory;


	public DefaultSessionFactoryImpl(MatrixFactory matrixFactory, DirectedComponentFactory directedComponentFactory, 
			DirectedLayerFactory directedLayerFactory, SupervisedFeedForwardNeuralNetworkFactory supervisedFeedForwardNeuralNetworkFactory,
			LayeredSupervisedFeedForwardNeuralNetworkFactory layeredSupervisedFeedForwardNeuralNetworkFactory) {
		this.directedComponentFactory = directedComponentFactory;
		this.matrixFactory = matrixFactory;
		this.directedLayerFactory = directedLayerFactory;
		this.supervisedFeedForwardNeuralNetworkFactory = supervisedFeedForwardNeuralNetworkFactory;
		this.layeredSupervisedFeedForwardNeuralNetworkFactory = layeredSupervisedFeedForwardNeuralNetworkFactory;
	}
	
	

	@Override
	public DefaultSession createSession(DirectedComponentsContext directedComponentsContext) {
		return new DefaultSessionImpl(directedComponentFactory, directedLayerFactory, 
				supervisedFeedForwardNeuralNetworkFactory, layeredSupervisedFeedForwardNeuralNetworkFactory, directedComponentsContext);
	}

	@Override
	public DirectedComponentFactory getNeuralComponentFactory() {
		return directedComponentFactory;
	}

	@Override
	public DefaultSession createSession() {
		return new DefaultSessionImpl(directedComponentFactory, directedLayerFactory, supervisedFeedForwardNeuralNetworkFactory, layeredSupervisedFeedForwardNeuralNetworkFactory, new DirectedComponentsContextImpl(matrixFactory, false));
	}
}
