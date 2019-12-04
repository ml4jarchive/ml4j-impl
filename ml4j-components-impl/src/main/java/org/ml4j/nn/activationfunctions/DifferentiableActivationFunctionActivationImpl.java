package org.ml4j.nn.activationfunctions;

import java.util.Arrays;
import java.util.List;

import org.ml4j.InterrimMatrix;
import org.ml4j.Matrix;
import org.ml4j.nn.components.ChainableDirectedComponentActivation;
import org.ml4j.nn.components.DirectedComponentGradient;
import org.ml4j.nn.components.DirectedComponentGradientImpl;
import org.ml4j.nn.costfunctions.CostFunctionGradient;
import org.ml4j.nn.neurons.NeuronsActivation;
import org.ml4j.nn.neurons.NeuronsActivationContext;
import org.ml4j.nn.neurons.NeuronsActivationImpl;

public class DifferentiableActivationFunctionActivationImpl implements DifferentiableActivationFunctionActivation {

	private NeuronsActivation input;
	private NeuronsActivation output;
	private DifferentiableActivationFunction activationFunction;
	private NeuronsActivationContext neuronsActivationContext;

	/**
	 * @param activationFunction
	 *            The activation function that generated this activation.
	 * @param input
	 *            The input.
	 * @param output
	 *            The output.
	 */
	public DifferentiableActivationFunctionActivationImpl(DifferentiableActivationFunction activationFunction,
			NeuronsActivation input, NeuronsActivation output, NeuronsActivationContext neuronsActivationContext) {
		this.input = input;
		this.output = output;
		this.activationFunction = activationFunction;
		this.neuronsActivationContext = neuronsActivationContext;
	}

	@Override
	public NeuronsActivation getInput() {
		return input;
	}

	@Override
	public NeuronsActivation getOutput() {
		return output;
	}

	@Override
	public DirectedComponentGradient<NeuronsActivation> backPropagate(DirectedComponentGradient<NeuronsActivation> da) {

		try (InterrimMatrix activationGradient = activationFunction.activationGradient(this, neuronsActivationContext).getActivations(neuronsActivationContext.getMatrixFactory()).asInterrimMatrix()) {
			try (InterrimMatrix daOutput = da.getOutput().getActivations(neuronsActivationContext.getMatrixFactory()).asInterrimMatrix()) {
				
				Matrix dz = daOutput.mul(activationGradient);

				return new DirectedComponentGradientImpl<>(da.getTotalTrainableAxonsGradients(),
						(new NeuronsActivationImpl(dz, da.getOutput().getFeatureOrientation(), false)));

			}
		}
	}

	@Override
	public DirectedComponentGradient<NeuronsActivation> backPropagate(CostFunctionGradient da) {
		return da.backPropagateThroughFinalActivationFunction(activationFunction);
	}

	@Override
	public DifferentiableActivationFunction getActivationFunction() {
		return activationFunction;
	}

	@Override
	public List<ChainableDirectedComponentActivation<NeuronsActivation>> decompose() {
		return Arrays.asList(this);
	}
}
