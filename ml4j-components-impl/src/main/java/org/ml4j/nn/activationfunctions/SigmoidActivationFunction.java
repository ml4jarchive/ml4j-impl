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

import org.ml4j.InterrimMatrix;
import org.ml4j.Matrix;
import org.ml4j.nn.neurons.NeuronsActivation;
import org.ml4j.nn.neurons.NeuronsActivationContext;
import org.ml4j.nn.neurons.NeuronsActivationImpl;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * The default Sigmoid Activation Function.
 * 
 * @author Michael Lavelle
 *
 */
public class SigmoidActivationFunction implements DifferentiableActivationFunction {
  
  /**
   * Default serialization id.
   */
  private static final long serialVersionUID = 1L;
  
  private static final Logger LOGGER = LoggerFactory.getLogger(SigmoidActivationFunction.class);

  @Override
  public DifferentiableActivationFunctionActivation activate(NeuronsActivation input, 
      NeuronsActivationContext context) {
    LOGGER.debug("Activating through SigmoidActivationFunction:" + input.getFeatureCount() 
    		+ ":" + 
    		+ input.getExampleCount() + ":" + input.getFeatureOrientation());

    
    
    Matrix sigmoidOfInputActivationsMatrix = 
        input.getActivations(context.getMatrixFactory()).sigmoid();
    return new DifferentiableActivationFunctionActivationImpl(this, input, 
        new NeuronsActivationImpl(sigmoidOfInputActivationsMatrix,
        input.getFeatureOrientation()), context);
  }

  @Override
  public NeuronsActivation activationGradient(DifferentiableActivationFunctionActivation 
      activationFunctionActivation,
      NeuronsActivationContext context) {
    
    LOGGER.debug("Performing sigmoid gradient of NeuronsActivation:");
  
    if (activationFunctionActivation.getActivationFunction() instanceof SigmoidActivationFunction) {
    	Matrix sigmoidOfActivationInput = activationFunctionActivation.getOutput()
          .getActivations(context.getMatrixFactory());
      
			try (InterrimMatrix sigmoidOfActivationInputSquared = sigmoidOfActivationInput.mul(sigmoidOfActivationInput)
					.asInterrimMatrix()) {

				Matrix gradientAtActivationInput = sigmoidOfActivationInput.sub(sigmoidOfActivationInputSquared);

				return new NeuronsActivationImpl(gradientAtActivationInput,
						activationFunctionActivation.getInput().getFeatureOrientation());
			}

		} else {

			Matrix activationInput = activationFunctionActivation.getInput().getActivations(context.getMatrixFactory());
			try (InterrimMatrix sigmoidOfActivationInput = activationInput.sigmoid().asInterrimMatrix()) {
	
				try (InterrimMatrix sigmoidOfActivationInputSquared = sigmoidOfActivationInput.mul(sigmoidOfActivationInput)
						.asInterrimMatrix()) {
	
					Matrix gradientAtActivationInput = sigmoidOfActivationInput.sub(sigmoidOfActivationInputSquared);
	
					return new NeuronsActivationImpl(gradientAtActivationInput,
							activationFunctionActivation.getInput().getFeatureOrientation());
				}
			}
    
    }
   
   
  }
}
