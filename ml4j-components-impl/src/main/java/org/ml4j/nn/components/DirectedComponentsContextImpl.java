package org.ml4j.nn.components;

import java.io.Serializable;
import java.util.HashMap;
import java.util.Map;
import java.util.function.Supplier;

import org.ml4j.MatrixFactory;

public class DirectedComponentsContextImpl implements DirectedComponentsContext {

	private Map<DirectedComponent<?, ?, ?>, ComponentContext<?>> contexts;
	private MatrixFactory matrixFactory;
	
	public DirectedComponentsContextImpl(MatrixFactory matrixFactory) {
		this.contexts = new HashMap<>();
		this.matrixFactory = matrixFactory;
	}
	
	@SuppressWarnings("unchecked")
	@Override
	public <C extends Serializable> C getContext(DirectedComponent<?, ?, C> component, Supplier<C> defaultContextSupplier) {
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

}
