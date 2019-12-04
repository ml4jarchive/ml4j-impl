/*
 * Copyright 2017 the original author or authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
 * in compliance with the License. You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software distributed under the License
 * is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
 * or implied. See the License for the specific language governing permissions and limitations under
 * the License.
 */

package org.ml4j.nn.activationfunctions;

import org.ml4j.EditableMatrix;
import org.ml4j.InterrimMatrix;
import org.ml4j.Matrix;
import org.ml4j.nn.neurons.NeuronsActivation;
import org.ml4j.nn.neurons.NeuronsActivationContext;
import org.ml4j.nn.neurons.NeuronsActivationFeatureOrientation;
import org.ml4j.nn.neurons.NeuronsActivationImpl;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * The default Sigmoid Activation Function.
 * 
 * @author Michael Lavelle
 *
 */
public class SoftmaxActivationFunction implements DifferentiableActivationFunction {

  /**
   * Default serialization id.
   */
  private static final long serialVersionUID = 1L;
  
  private static final Logger LOGGER = LoggerFactory.getLogger(SoftmaxActivationFunction.class);

  @Override
  public DifferentiableActivationFunctionActivation activate(NeuronsActivation input, 
      NeuronsActivationContext context) {
    LOGGER.debug("Activating through SoftmaxActivationFunction");
        
    if (input
	        .getFeatureOrientation() != NeuronsActivationFeatureOrientation.ROWS_SPAN_FEATURE_SET) {
	      throw new IllegalArgumentException("Only neurons actiavation with ROWS_SPAN_FEATURE_SET "
	          + "orientation supported currently");
	    }
  
   
    Matrix softmaxOfInputActivationsMatrix = softmax(input.getActivations(context.getMatrixFactory()));
    return new DifferentiableActivationFunctionActivationImpl(this, input, 
        new NeuronsActivationImpl(softmaxOfInputActivationsMatrix,
        input.getFeatureOrientation()), context);
  }
  

  /**
   * Returns a matrix that has the softmax function applied to each element of given input matrix.
   */
  public static Matrix softmax(Matrix x1) {
	EditableMatrix exp = x1.asEditableMatrix().expi();
    try (InterrimMatrix sums = exp.columnSums().asInterrimMatrix()) {
    	return exp.diviRowVector(sums);
    }   
  }

  @Override
  public NeuronsActivation activationGradient(DifferentiableActivationFunctionActivation 
      outputActivation,
      NeuronsActivationContext context) {
    throw new UnsupportedOperationException("Standalone activation gradient of "
        + "softmax not supported - gradient calculation is performed in combination with a "
        + "specified cost function at the outer layer");
  }
}
