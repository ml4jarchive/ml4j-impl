package org.ml4j.nn.activationfunctions;

public class DefaultDifferentiableActivationFunctionFactory implements DifferentiableActivationFunctionFactory {

	@Override
	public DifferentiableActivationFunction createReluActivationFunction() {
		return new ReluActivationFunction();
	}

	@Override
	public DifferentiableActivationFunction createSigmoidActivationFunction() {
		return new SigmoidActivationFunction();
	}

	@Override
	public DifferentiableActivationFunction createSoftmaxActivationFunction() {
		return new SoftmaxActivationFunction();
	}

}
