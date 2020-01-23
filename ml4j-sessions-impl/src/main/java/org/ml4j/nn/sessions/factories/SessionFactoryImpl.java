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
import org.ml4j.nn.components.NeuralComponent;
import org.ml4j.nn.components.factories.NeuralComponentFactory;
import org.ml4j.nn.sessions.Session;
import org.ml4j.nn.sessions.SessionImpl;

/**
 * Factory implementation for Sessions
 * 
 * @author Michael Lavelle
 *
 * @param <T> The type of NeuralComponent within the Session.
 */
public class SessionFactoryImpl<T extends NeuralComponent> implements SessionFactory<T> {

	private NeuralComponentFactory<T> neuralComponentFactory;
	private MatrixFactory matrixFactory;

	public SessionFactoryImpl(MatrixFactory matrixFactory, NeuralComponentFactory<T> neuralComponentFactory) {
		this.neuralComponentFactory = neuralComponentFactory;
		this.matrixFactory = matrixFactory;
	}

	@Override
	public Session<T> createSession(DirectedComponentsContext directedComponentsContext) {
		return new SessionImpl<T>(neuralComponentFactory, directedComponentsContext);
	}

	@Override
	public NeuralComponentFactory<T> getNeuralComponentFactory() {
		return neuralComponentFactory;
	}

	@Override
	public Session<T> createSession() {
		return new SessionImpl<T>(neuralComponentFactory, new DirectedComponentsContextImpl(matrixFactory, false));
	}

}
