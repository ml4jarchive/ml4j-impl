package org.ml4j.nn.components.defaults;

import java.util.List;

import org.ml4j.nn.components.DirectedComponent;
import org.ml4j.nn.components.DirectedComponentActivation;
import org.ml4j.nn.components.DirectedComponentBatchImpl;
import org.ml4j.nn.components.DirectedComponentsContext;
import org.ml4j.nn.neurons.NeuronsActivation;

public class DefaultDirectedComponentBatchImpl<L extends DirectedComponent<NeuronsActivation, A, DirectedComponentsContext> , A extends  DirectedComponentActivation<NeuronsActivation, NeuronsActivation>> extends DirectedComponentBatchImpl<NeuronsActivation, L, A, DirectedComponentsContext, DirectedComponentsContext> {

	/**
	 * Defualt serialziation id.
	 */
	private static final long serialVersionUID = 1L;

	public DefaultDirectedComponentBatchImpl(List<L> components) {
		super(components);
	}

	@Override
	protected DirectedComponentsContext getContext(DirectedComponentsContext context, L component, int index) {
		return context;
	}

	

}
