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
package org.ml4j.nn.sessions;

import org.ml4j.MatrixFactory;
import org.ml4j.nn.components.DirectedComponentsContext;
import org.ml4j.nn.components.NeuralComponent;
import org.ml4j.nn.components.factories.NeuralComponentFactory;

/**
 * Default Session implementation for the creation of Neural Component graphs.
 * 
 * @author Michael Lavelle
 *
 * @param <T> The type of NeuralComponent within the session.
 */
public class SessionImpl<T extends NeuralComponent> implements Session<T> {

	private NeuralComponentFactory<T> neuralComponentFactory;
	private DirectedComponentsContext directedComponentsContext;

	public SessionImpl(NeuralComponentFactory<T> neuralComponentFactory,
			DirectedComponentsContext directedComponentsContext) {
		this.neuralComponentFactory = neuralComponentFactory;
		this.directedComponentsContext = directedComponentsContext;
	}

	@Override
	public NeuralComponentFactory<T> getNeuralComponentFactory() {
		return neuralComponentFactory;
	}

	@Override
	public ComponentGraphBuilderSession<T> buildComponentGraph() {
		return new ComponentGraphBuilderSessionImpl<>(neuralComponentFactory, directedComponentsContext);
	}

	@Override
	public DirectedComponentsContext getDirectedComponentsContext() {
		return directedComponentsContext;
	}

	@Override
	public NeuralNetworkBuilderSession<T> buildNeuralNetwork() {
		throw new UnsupportedOperationException("Not yet implemented");
	}

	@Override
	public MatrixFactory getMatrixFactory() {
		return directedComponentsContext.getMatrixFactory();
	}

}
