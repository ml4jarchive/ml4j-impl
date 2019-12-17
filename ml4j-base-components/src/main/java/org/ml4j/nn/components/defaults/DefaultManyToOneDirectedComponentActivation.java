package org.ml4j.nn.components.defaults;

import org.ml4j.nn.components.GenericManyToOneDirectedComponentActivationImpl;
import org.ml4j.nn.components.ManyToOneDirectedComponentActivation;
import org.ml4j.nn.neurons.NeuronsActivation;

public class DefaultManyToOneDirectedComponentActivation extends GenericManyToOneDirectedComponentActivationImpl<NeuronsActivation>
		implements ManyToOneDirectedComponentActivation {

	public DefaultManyToOneDirectedComponentActivation(NeuronsActivation output, int inputCount) {
		super(output, inputCount);
	}
}
