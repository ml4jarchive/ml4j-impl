package org.ml4j.nn.components.defaults;

import java.util.List;

import org.ml4j.nn.components.ChainableDirectedComponent;
import org.ml4j.nn.components.ChainableDirectedComponentActivation;
import org.ml4j.nn.components.DirectedComponentsContext;
import org.ml4j.nn.components.OneToManyDirectedComponent;
import org.ml4j.nn.components.OneToManyDirectedComponentActivation;
import org.ml4j.nn.neurons.NeuronsActivation;

public class DefaultOneToManyDirectedComponent extends OneToManyDirectedComponent<NeuronsActivation, DirectedComponentsContext> {

	/**
	 * Default serialization id
	 */
	private static final long serialVersionUID = 1L;

	public DefaultOneToManyDirectedComponent(List<? extends ChainableDirectedComponent<NeuronsActivation, 
			? extends ChainableDirectedComponentActivation<NeuronsActivation>, DirectedComponentsContext>> targetComponents) {
		super(targetComponents);
	}

	@Override
	protected OneToManyDirectedComponentActivation<NeuronsActivation> createActivation(List<NeuronsActivation> acts,
			DirectedComponentsContext synapsesContext) {
		return new DefaultOneToManyDirectedComponentActivation(acts, synapsesContext.getMatrixFactory());
	}

	@Override
	public OneToManyDirectedComponentActivation<NeuronsActivation> forwardPropagate(NeuronsActivation input,
			DirectedComponentsContext synapsesContext) {
		input.setImmutable(true);
		return super.forwardPropagate(input, synapsesContext);
	}
	
	
}
