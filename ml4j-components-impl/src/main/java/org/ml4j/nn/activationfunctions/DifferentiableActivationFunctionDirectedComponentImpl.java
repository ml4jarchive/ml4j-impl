package org.ml4j.nn.activationfunctions;

import java.util.Arrays;
import java.util.List;

import org.ml4j.MatrixFactory;
import org.ml4j.nn.components.ChainableDirectedComponent;
import org.ml4j.nn.components.ChainableDirectedComponentActivation;
import org.ml4j.nn.components.DirectedComponentsContext;
import org.ml4j.nn.neurons.NeuronsActivation;
import org.ml4j.nn.neurons.NeuronsActivationContext;

public class DifferentiableActivationFunctionDirectedComponentImpl implements DifferentiableActivationFunctionComponent {

	/**
	 * Default serialization id.
	 */
	private static final long serialVersionUID = 1L;
	
	private DifferentiableActivationFunction activationFunction;
	
	public DifferentiableActivationFunctionDirectedComponentImpl(DifferentiableActivationFunction activationFunction) {
		this.activationFunction = activationFunction;
	}

	@Override
	public DifferentiableActivationFunctionActivation forwardPropagate(NeuronsActivation input,
			NeuronsActivationContext synapsesContext) {
		return activationFunction.activate(input, synapsesContext);
	}

	@Override
	public DifferentiableActivationFunction getActivationFunction() {
		return activationFunction;
	}

	@Override
	public NeuronsActivationContext getContext(DirectedComponentsContext directedComponentsContext, int componentIndex) {
		NeuronsActivationContext context = new NeuronsActivationContext() {

			/**
			 * Default serialization id.
			 */
			private static final long serialVersionUID = 1L;

			@Override
			public MatrixFactory getMatrixFactory() {
				return directedComponentsContext.getMatrixFactory();
			};
		};
		return directedComponentsContext.getContext(this, () -> context);
	}

	@Override
	public List<ChainableDirectedComponent<NeuronsActivation, ? extends ChainableDirectedComponentActivation<NeuronsActivation>, ?>> decompose() {
		return Arrays.asList(this);
	}

}
