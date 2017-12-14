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
import org.ml4j.nn.activationfunctions.ActivationFunctionGradient;
import org.ml4j.nn.activationfunctions.ActivationFunctionGradientImpl;
import org.ml4j.nn.activationfunctions.DifferentiableActivationFunction;
import org.ml4j.nn.activationfunctions.LinearActivationFunction;
import org.ml4j.nn.activationfunctions.SigmoidActivationFunction;
import org.ml4j.nn.activationfunctions.SoftmaxActivationFunction;
import org.ml4j.nn.neurons.NeuronsActivation;
import org.ml4j.nn.neurons.NeuronsActivationFeatureOrientation;

/**
 * Responsible for back propagating the CostFunctionGradient through a final activation function
 * compatible with the cost function, by using the delta rule.
 * 
 * @author Michael Lavelle
 */
public class DeltaRuleCostFunctionGradientImpl implements CostFunctionGradient {

  private NeuronsActivation desiredOutputs;
  private NeuronsActivation actualOutputs;
  private CostFunction costFunction;

  /**
   * @param costFunction The cost function.
   * @param desiredOutputs The desired outputs of the network.
   * @param actualOutputs The actual outputs of the network.
   */
  public DeltaRuleCostFunctionGradientImpl(CostFunction costFunction,
      NeuronsActivation desiredOutputs, NeuronsActivation actualOutputs) {
    this.desiredOutputs = desiredOutputs;
    this.costFunction = costFunction;
    this.actualOutputs = actualOutputs;
  }

  private boolean isDeltaRuleSupported(DifferentiableActivationFunction finalActivationFunction) {

    if (costFunction instanceof CrossEntropyCostFunction
        && finalActivationFunction instanceof SigmoidActivationFunction) {
      return true;
    }
    if (costFunction instanceof MultiClassCrossEntropyCostFunction
        && finalActivationFunction instanceof SoftmaxActivationFunction) {
      return true;
    }
    if (costFunction instanceof SumSquaredErrorCostFunction
        && finalActivationFunction instanceof LinearActivationFunction) {
      return true;
    }
    return false;
  }

  @Override
  public ActivationFunctionGradient backPropagateThroughFinalActivationFunction(
      DifferentiableActivationFunction finalActivationFunction) {

    if (!isDeltaRuleSupported(finalActivationFunction)) {
      throw new IllegalArgumentException(
          "Activation Function " + finalActivationFunction.getClass().getName()
              + " not supported for the delta rule for cost function:"
              + costFunction.getClass().getName());
    }


    // When using either of the cross entropy cross functions,
    // the deltas we backpropagate
    // end up being the difference between the target activations ( which are the
    // same as the trainingDataActivations as this is an AutoEncoder), and the
    // activations resulting from the forward propagation
    Matrix deltasM = actualOutputs.getActivations().sub(desiredOutputs.getActivations());

    // The deltas we back propagate are in the transposed orientation to the inputs
    NeuronsActivation deltas = new NeuronsActivation(deltasM.transpose(),
        NeuronsActivationFeatureOrientation.ROWS_SPAN_FEATURE_SET);

    return new ActivationFunctionGradientImpl(deltas);

  }

  @Override
  public CostFunction getCostFunction() {
    return costFunction;
  }
}
