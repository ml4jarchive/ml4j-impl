package org.ml4j.nn.activationfunctions.mocks;

import org.ml4j.nn.activationfunctions.ActivationFunctionType;
import org.ml4j.nn.activationfunctions.DifferentiableActivationFunction;
import org.ml4j.nn.activationfunctions.factories.DifferentiableActivationFunctionFactory;

public class DummyDifferentiableActivationFunctionFactory implements DifferentiableActivationFunctionFactory {

	@Override
	public DifferentiableActivationFunction createReluActivationFunction() {
		return new DummyDifferentiableActivationFunction(ActivationFunctionType.RELU, false);
	}

	@Override
	public DifferentiableActivationFunction createSigmoidActivationFunction() {
		return new DummyDifferentiableActivationFunction(ActivationFunctionType.SIGMOID, false);
	}

	@Override
	public DifferentiableActivationFunction createSoftmaxActivationFunction() {
		return new DummyDifferentiableActivationFunction(ActivationFunctionType.SOFTMAX, true);
	}

	@Override
	public DifferentiableActivationFunction createLinearActivationFunction() {
		return new DummyDifferentiableActivationFunction(ActivationFunctionType.LINEAR, true);
	}

}
