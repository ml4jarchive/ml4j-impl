package org.ml4j.nn.components;

import org.ml4j.nn.neurons.NeuronsActivationContextImpl;

public class DirectedComponentActivationContextImpl extends NeuronsActivationContextImpl implements DirectedComponentActivationContext {

	/**
	 * Default serialization id.
	 */
	private static final long serialVersionUID = 1L;

	private transient DirectedComponentsContext directedComponentsContext;
	
	public DirectedComponentActivationContextImpl(DirectedComponentsContext directedComponentsContext) {
		super(directedComponentsContext.getMatrixFactory(), directedComponentsContext.isTrainingContext());
		this.directedComponentsContext = directedComponentsContext;
	}

	@Override
	public DirectedComponentsContext getDirectedComponentsContext() {
		return directedComponentsContext;
	}

	@Override
	public void setDirectedComponentsContext(DirectedComponentsContext directedComponentsContext) {
		this.directedComponentsContext = directedComponentsContext;
	}

}
