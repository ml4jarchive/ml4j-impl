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
import java.util.Optional;
import java.util.function.Supplier;
import java.util.function.UnaryOperator;

import org.ml4j.MatrixFactory;
import org.ml4j.nn.neurons.NeuronsActivationContext;

public class DirectedComponentsContextImpl implements DirectedComponentsContext {

	/**
	 * Default serialization id.
	 */
	private static final long serialVersionUID = 1L;
	private Map<ContextualNeuralComponent<?>, ComponentContext<?>> contexts;
	private MatrixFactory matrixFactory;
	private boolean isTraining;

	public DirectedComponentsContextImpl(MatrixFactory matrixFactory, boolean isTraining) {
		this.contexts = new HashMap<>();
		this.matrixFactory = matrixFactory;
		this.isTraining = isTraining;
	}

	private DirectedComponentsContextImpl(Map<ContextualNeuralComponent<?>, ComponentContext<?>> contexts,
			MatrixFactory matrixFactory, boolean isTraining) {
		super();
		this.contexts = contexts;
		this.matrixFactory = matrixFactory;
		this.isTraining = isTraining;
	}
	
	private boolean isComponentNameExistingUnderAnotherComponent(ContextualNeuralComponent<?> component) {
		Optional<ContextualNeuralComponent<?>> found = contexts.keySet().stream()
				.filter(c -> c.getName().equals(component.getName())).findFirst();
		return found.isPresent();
	}
	
	private <C extends Serializable> Object getLockObject(ContextualNeuralComponent<C> component) {
		synchronized (contexts) {
			ComponentContext<?> componentContext = contexts.get(component);
			if (componentContext == null) {
				return contexts;
			} else {
				return componentContext;
			}
		}
	}
	
	@SuppressWarnings("unchecked")
	@Override
	public  <C extends Serializable> C getContext(ContextualNeuralComponent<C> component,
			Supplier<C> defaultContextSupplier, UnaryOperator<C> creator) {
	
		synchronized (getLockObject(component)) {
		
			ComponentContext<C> existingContext = (ComponentContext<C>) contexts.get(component);
	
			C context;
			if (existingContext != null) {
				context = creator.apply(existingContext.getContext());
			} else {
				ComponentContext<C> newContext = new ComponentContext<>(component, defaultContextSupplier.get());
				contexts.put(component, newContext);
				context = newContext.getContext();
			}
			
			if (context instanceof NeuronsActivationContext) {
				NeuronsActivationContext neuronsActivationContext = (NeuronsActivationContext)context;
				neuronsActivationContext.setTrainingContext(isTraining);
				neuronsActivationContext.setMatrixFactory(getMatrixFactory());
			}
			return context;
		}
	}

	public void addComponentContext(ComponentContext<?> componentContext) {
		
		synchronized (getLockObject(componentContext.getComponent())) {
		
			if (isComponentNameExistingUnderAnotherComponent(componentContext.getComponent())) {
				throw new IllegalArgumentException("Component name already registered under another component:" + 
						componentContext.getComponent().getName());
			}
			this.contexts.put(componentContext.getComponent(), componentContext);
		}
	}

	private class ComponentContext<C extends Serializable> implements Serializable {

		/**
		 * Default serialization id.
		 */
		private static final long serialVersionUID = 1L;
		private ContextualNeuralComponent<C> component;
		private C context;

		public ComponentContext(ContextualNeuralComponent<C> component, C context) {
			this.component = component;
			this.context = context;
		}

		public ContextualNeuralComponent<C> getComponent() {
			return component;
		}

		public C getContext() {
			return context;
		}

		@Override
		public String toString() {
			return "ComponentContext [component=" + component.getName() + ", context=" + context + "]";
		}
		
		

	}

	@Override
	public MatrixFactory getMatrixFactory() {
		return matrixFactory;
	}

	@Override
	public <C extends Serializable> void setContext(ContextualNeuralComponent<C> component, C context) {
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

	@Override
	public String toString() {
		return "DirectedComponentsContextImpl [contexts=" + contexts.values() + ", matrixFactory=" + matrixFactory
				+ ", isTraining=" + isTraining + "]";
	}
	
	
}
