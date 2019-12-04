package org.ml4j.nn.activationfunctions;

import org.ml4j.EditableMatrix;
import org.ml4j.InterrimMatrix;
import org.ml4j.nn.neurons.NeuronsActivation;
import org.ml4j.nn.neurons.NeuronsActivationContext;
import org.ml4j.nn.neurons.NeuronsActivationImpl;

public class BernoulliOfSigmoidActivationFunction implements DifferentiableActivationFunction {

	/**
	 * Default serialization id.
	 */
	private static final long serialVersionUID = 1L;

	private SigmoidActivationFunction sigmoidActivationFunction;

	public BernoulliOfSigmoidActivationFunction() {
		this.sigmoidActivationFunction = new SigmoidActivationFunction();
	}

	@Override
	public DifferentiableActivationFunctionActivation activate(NeuronsActivation input,
			NeuronsActivationContext context) {
		try (InterrimMatrix random = context.getMatrixFactory()
				.createRand(input.getActivations(context.getMatrixFactory()).getRows(), input.getActivations(context.getMatrixFactory()).getColumns()).asInterrimMatrix()) {
			EditableMatrix activation = sigmoidActivationFunction.activate(input, context).getOutput().getActivations(context.getMatrixFactory()).asEditableMatrix();
			for (int r = 0; r < random.getRows(); r++) {
				for (int c = 0; c < random.getColumns(); c++) {
					if (random.get(r, c) < activation.get(r, c)) {
						activation.put(r, c, 1);
					} else {
						activation.put(r, c, 0);
					}
				}
			}
			return new DifferentiableActivationFunctionActivationImpl(this, input, 
					new NeuronsActivationImpl(activation, input.getFeatureOrientation()), context);
		}
	}

	@Override
	public NeuronsActivation activationGradient(DifferentiableActivationFunctionActivation outputActivation,
			NeuronsActivationContext context) {
		return sigmoidActivationFunction.activationGradient(outputActivation, context);
	}

}
