/*
 * Copyright 2017 the original author or authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.ml4j.nn.costfunctions;

import org.ml4j.Matrix;
import org.ml4j.MatrixFactory;
import org.ml4j.nn.activationfunctions.ActivationFunctionBaseType;
import org.ml4j.nn.activationfunctions.ActivationFunctionType;
import org.ml4j.nn.components.DirectedComponentGradient;
import org.ml4j.nn.components.DirectedComponentGradientImpl;
import org.ml4j.nn.neurons.NeuronsActivation;
import org.ml4j.nn.neurons.NeuronsActivationFeatureOrientation;
import org.ml4j.nn.neurons.NeuronsActivationImpl;

/**
 * Responsible for back propagating the CostFunctionGradient through a final
 * activation function compatible with the cost function, by using the delta
 * rule.
 * 
 * @author Michael Lavelle
 */
public class DeltaRuleCostFunctionGradientImpl implements CostFunctionGradient {

	private NeuronsActivation desiredOutputs;
	private NeuronsActivation actualOutputs;
	private CostFunction costFunction;
	private MatrixFactory matrixFactory;

	/**
	 * @param costFunction   The cost function.
	 * @param desiredOutputs The desired outputs of the network.
	 * @param actualOutputs  The actual outputs of the network.
	 */
	public DeltaRuleCostFunctionGradientImpl(MatrixFactory matrixFactory, CostFunction costFunction,
			NeuronsActivation desiredOutputs, NeuronsActivation actualOutputs) {
		this.desiredOutputs = desiredOutputs;
		this.matrixFactory = matrixFactory;

		if (desiredOutputs.getFeatureOrientation() != NeuronsActivationFeatureOrientation.ROWS_SPAN_FEATURE_SET) {
			throw new IllegalArgumentException(
					"Only neurons actiavation with ROWS_SPAN_FEATURE_SET " + "orientation supported currently");
		}
		if (actualOutputs.getFeatureOrientation() != NeuronsActivationFeatureOrientation.ROWS_SPAN_FEATURE_SET) {
			throw new IllegalArgumentException(
					"Only neurons actiavation with ROWS_SPAN_FEATURE_SET " + "orientation supported currently");
		}
		if (actualOutputs.getColumns() != desiredOutputs.getColumns()) {
			throw new IllegalArgumentException("Mismatched column count between desired and actual outputs");
		}
		if (actualOutputs.getRows() != desiredOutputs.getRows()) {
			throw new IllegalArgumentException("Mismatched row count between desired and actual outputs");
		}
		this.costFunction = costFunction;
		this.actualOutputs = actualOutputs;
	}

	private boolean isDeltaRuleSupported(ActivationFunctionType finalActivationFunctionType) {

		if (costFunction instanceof CrossEntropyCostFunction
				&& ActivationFunctionBaseType.SIGMOID.equals(finalActivationFunctionType.getBaseType())) {
			return true;
		}
		if (costFunction instanceof MultiClassCrossEntropyCostFunction
				&& ActivationFunctionBaseType.SOFTMAX.equals(finalActivationFunctionType.getBaseType())) {
			return true;
		}
		if (costFunction instanceof SumSquaredErrorCostFunction
				&& ActivationFunctionBaseType.LINEAR.equals(finalActivationFunctionType.getBaseType())) {
			return true;
		}
		return false;
	}

	@Override
	public DirectedComponentGradient<NeuronsActivation> backPropagateThroughFinalActivationFunction(
			ActivationFunctionType finalActivationFunctionType) {

		if (!isDeltaRuleSupported(finalActivationFunctionType)) {
			throw new IllegalArgumentException("Activation Function " + finalActivationFunctionType
					+ " not supported for the delta rule for cost function:" + costFunction.getClass().getName());
		}

		// When using either of the cross entropy cross functions,
		// the deltas we backpropagate
		// end up being the difference between the target activations ( which are the
		// same as the trainingDataActivations as this is an AutoEncoder), and the
		// activations resulting from the forward propagation

		Matrix deltasM = actualOutputs.getActivations(matrixFactory).sub(desiredOutputs.getActivations(matrixFactory));

		actualOutputs.getActivations(matrixFactory).close();

		NeuronsActivation deltas = new NeuronsActivationImpl(actualOutputs.getNeurons(), deltasM,
				actualOutputs.getFormat());

		return new DirectedComponentGradientImpl<>(deltas);

	}

	@Override
	public CostFunction getCostFunction() {
		return costFunction;
	}

}
