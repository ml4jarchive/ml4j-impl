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
package org.ml4j.nn.components;

import java.io.Serializable;
import java.util.HashMap;
import java.util.Map;
import java.util.function.Supplier;

import org.ml4j.MatrixFactory;

public class DirectedComponentsContextImpl implements DirectedComponentsContext {

	/**
	 * Default serialization id.
	 */
	private static final long serialVersionUID = 1L;
	private Map<DirectedComponent<?, ?, ?>, ComponentContext<?>> contexts;
	private MatrixFactory matrixFactory;
	private boolean isTraining;

	public DirectedComponentsContextImpl(MatrixFactory matrixFactory, boolean isTraining) {
		this.contexts = new HashMap<>();
		this.matrixFactory = matrixFactory;
		this.isTraining = isTraining;
	}

	private DirectedComponentsContextImpl(Map<DirectedComponent<?, ?, ?>, ComponentContext<?>> contexts,
			MatrixFactory matrixFactory, boolean isTraining) {
		super();
		this.contexts = contexts;
		this.matrixFactory = matrixFactory;
		this.isTraining = isTraining;
	}
	
	@SuppressWarnings("unchecked")
	@Override
	public <C extends Serializable> C getContext(DirectedComponent<?, ?, C> component,
			Supplier<C> defaultContextSupplier) {
		ComponentContext<C> existingContext = (ComponentContext<C>) contexts.get(component);

		if (existingContext != null) {
			return existingContext.getContext();
		} else {
			ComponentContext<C> newContext = new ComponentContext<>(component, defaultContextSupplier.get());
			contexts.put(component, newContext);
			return newContext.getContext();
		}
	}

	public void addComponentContext(ComponentContext<?> componentContext) {
		this.contexts.put(componentContext.getComponent(), componentContext);
	}

	private class ComponentContext<C extends Serializable> implements Serializable {

		/**
		 * Default serialization id.
		 */
		private static final long serialVersionUID = 1L;
		private DirectedComponent<?, ?, C> component;
		private C context;

		public ComponentContext(DirectedComponent<?, ?, C> component, C context) {
			this.component = component;
			this.context = context;
		}

		public DirectedComponent<?, ?, C> getComponent() {
			return component;
		}

		public C getContext() {
			return context;
		}

	}

	@Override
	public MatrixFactory getMatrixFactory() {
		return matrixFactory;
	}

	@Override
	public <C extends Serializable> void setContext(DirectedComponent<?, ?, C> component, C context) {
		this.contexts.put(component, new ComponentContext<C>(component, context));
	}

	@Override
	public boolean isTrainingContext() {
		return isTraining;
	}

	@Override
	public DirectedComponentsContext asTrainingContext() {
		return new DirectedComponentsContextImpl(contexts, matrixFactory, true);
	}

	@Override
	public DirectedComponentsContext asNonTrainingContext() {
		return new DirectedComponentsContextImpl(contexts, matrixFactory, false);
	}
}
