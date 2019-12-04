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

import org.ml4j.Matrix;
import org.ml4j.nn.neurons.NeuronsActivation;
import org.ml4j.nn.neurons.NeuronsActivationContext;
import org.ml4j.nn.neurons.NeuronsActivationImpl;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Naive implementation of Rectified Linear Unit Activation Function 
 * (ReLU Activation Function).
 * 
 * <p>The ReluActivationFunction is not actually differentiable at 0, but is psedo-differentiable
 * for the purposes of use as part of back propagation.  We define that the gradient is 0
 * at 0 in this function.
 * 
 * @author Michael Lavelle
 *
 */
public class ReluActivationFunction implements DifferentiableActivationFunction {
  /**
   * Default serialization id.
   */
  private static final long serialVersionUID = 1L;
  
  private static final Logger LOGGER = LoggerFactory.getLogger(ReluActivationFunction.class);

  @Override
  public DifferentiableActivationFunctionActivation activate(NeuronsActivation input, 
      NeuronsActivationContext context) {
    LOGGER.debug("Activating through ReluActivationFunction");
    
    //Matrix outputMatrix = input.getActualActivations( context.getMatrixFactory());
    
    NeuronsActivation output  = input.dup();
  
    output.applyValueModifier(v -> v < 0, v -> 0);
    /*
    for (int r = 0; r < outputMatrix.getRows(); r++) {
      for (int c = 0; c < outputMatrix.getColumns(); c++) {
        if (outputMatrix.get(r, c) < 0) {
        	outputMatrix.put(r, c, 0f);
        }
      }
    }
   	*/
    /*
    NeuronsActivation outputAct = null;
    if (input instanceof NeuronsActivationCustom) {
    	Neurons3D neurons = ((NeuronsActivationCustom)input).getNeurons();
    	outputAct = new NeuronsActivationCustom(outputMatrix, neurons, input.getFeatureOrientation());
    } else {
    	outputAct = new NeuronsActivation(outputMatrix, input.getFeatureOrientation());
    }
    */
    
    return new DifferentiableActivationFunctionActivationImpl(this, input, output, context);
  }

  @Override
  public NeuronsActivation activationGradient(
      DifferentiableActivationFunctionActivation outputActivation,
      NeuronsActivationContext context) {
   
    LOGGER.debug("Performing relu gradient of NeuronsActivation");
    
    Matrix in = outputActivation.getInput().getActivations(context.getMatrixFactory());
    
    Matrix gradientMatrix = context.getMatrixFactory()
        .createOnes(in.getRows(), 
        		in.getColumns());
    
    
    NeuronsActivation output = new NeuronsActivationImpl(gradientMatrix, 
        outputActivation.getInput().getFeatureOrientation());
    
    // TODO
    //EditableMatrix editableActivations = output.getActivations(context.getMatrixFactory()).asEditableMatrix();
    output.applyValueModifier(v -> v <= 0, v -> 0);
    /*
    for (int r = 0; r < output.getActivations(context.getMatrixFactory()).getRows(); r++) {
      for (int c = 0; c < output.getActivations(context.getMatrixFactory()).getColumns(); c++) {

        if (in.get(r, c) <= 0) {
        	editableActivations.put(r, c, 0f);
        } 
      }
    }
    */

    return output;
  }
}
