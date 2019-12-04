package org.ml4j.nn.components.defaults;

import java.util.List;

import org.ml4j.nn.components.DirectedComponentsContext;
import org.ml4j.nn.components.ManyToOneDirectedComponent;
import org.ml4j.nn.components.ManyToOneDirectedComponentActivation;
import org.ml4j.nn.neurons.NeuronsActivation;
import org.ml4j.nn.neurons.NeuronsActivationImpl;

public class DefaultManyToOneAdditionDirectedComponent extends ManyToOneDirectedComponent<NeuronsActivation, DirectedComponentsContext> {

	/**
	 * Default serialization id.
	 */
	private static final long serialVersionUID = 1L;

	@Override
	protected NeuronsActivation getCombinedOutput(List<NeuronsActivation> gradient, DirectedComponentsContext context) {
		NeuronsActivation totalActivations = 
				new NeuronsActivationImpl(context.getMatrixFactory().createMatrix(gradient.get(0).getActivations(context.getMatrixFactory()).getRows(), 
						gradient.get(0).getActivations(context.getMatrixFactory()).getColumns()), gradient.get(0).getFeatureOrientation());
		for (NeuronsActivation activation : gradient) {
			totalActivations.addInline(context.getMatrixFactory(), activation);
		}
		return totalActivations;
	}
	@Override
	protected  ManyToOneDirectedComponentActivation<NeuronsActivation> createActivation(NeuronsActivation combinedInput, List<NeuronsActivation> input) {
		combinedInput.setImmutable(true);
		return new ManyToOneDirectedComponentActivation<>(combinedInput, input.size());
	}

}
