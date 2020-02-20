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

package org.ml4j.nn.axons;

import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;
import org.ml4j.Matrix;
import org.ml4j.MatrixFactory;
import org.ml4j.jblas.JBlasRowMajorMatrixFactory;
import org.ml4j.nn.neurons.Neurons;
import org.ml4j.nn.neurons.NeuronsActivation;
import org.ml4j.nn.neurons.NeuronsActivationFeatureOrientation;
import org.ml4j.nn.neurons.NeuronsActivationImpl;
import org.ml4j.nn.neurons.format.NeuronsActivationFormat;

public abstract class ScaleAndShiftAxonsTestBase {

  protected MatrixFactory matrixFactory;

  @Before
  public void setUp() {
    matrixFactory = new JBlasRowMajorMatrixFactory();
  }

  protected abstract ScaleAndShiftAxons<?> createAxons(Neurons leftNeurons, Neurons rightNeurons,
      ScaleAndShiftAxonsConfig config);

  @Test(expected=IllegalArgumentException.class)
  public void testPushLeftToRightWithNoLeftHandBias() {

    int featureCount = 10;
    Matrix scaleRowVector = matrixFactory.createRandn(1, featureCount);
    Matrix shiftRowVector = matrixFactory.createRandn(1, featureCount);
    Neurons leftNeurons = new Neurons(featureCount, false);
    Neurons rightNeurons = new Neurons(featureCount, false);
    ScaleAndShiftAxonsConfig config = new ScaleAndShiftAxonsConfig(leftNeurons, rightNeurons, scaleRowVector, shiftRowVector);

    createAxons(leftNeurons, rightNeurons, config);
  }
  
  @Test
  public void testPushLeftToRightWithLeftHandBias() {

    int featureCount = 10;
    Matrix scaleColumnVector = matrixFactory.createRandn(featureCount, 1);
    Matrix shiftColumnVector = matrixFactory.createRandn(featureCount, 1);
    Neurons leftNeurons = new Neurons(featureCount, true);
    Neurons rightNeurons = new Neurons(featureCount, false);
    ScaleAndShiftAxonsConfig config = new ScaleAndShiftAxonsConfig(leftNeurons, rightNeurons, scaleColumnVector, shiftColumnVector);

    ScaleAndShiftAxons<?> axons = createAxons(leftNeurons, rightNeurons, config);

    Matrix inputMatrix = matrixFactory.createRand(featureCount, 100);
 
    Assert.assertEquals(featureCount, inputMatrix.getRows());
    Assert.assertEquals(100, inputMatrix.getColumns());

    NeuronsActivation input = new NeuronsActivationImpl(leftNeurons, inputMatrix,
    		NeuronsActivationFormat.ROWS_SPAN_FEATURE_SET);
    AxonsContext context = new AxonsContextImpl("someAxons", matrixFactory, false, false);
    AxonsActivation axonsActivation = axons.pushLeftToRight(input, null, context);

    Matrix outputMatrix = axonsActivation.getPostDropoutOutput().getActivations(matrixFactory);

    Assert.assertEquals(inputMatrix.getRows(), outputMatrix.getRows());

    Assert.assertEquals(inputMatrix.getColumns(), outputMatrix.getColumns());

    for (int r = 0; r < outputMatrix.getRows(); r++) {
      for (int c = 0; c < outputMatrix.getColumns(); c++) {

        double scale = scaleColumnVector.get(r, 0);
        double shift = shiftColumnVector.get(r, 0);

        Assert.assertEquals(inputMatrix.get(r, c) * scale + shift, outputMatrix.get(r, c),
            0.000001);
      }
    }
  }
  
  @Test(expected=IllegalArgumentException.class)
  public void testPushLeftToRightWithLeftHandAndRightHandBias() {

    int featureCount = 10;
    Matrix scaleRowVector = matrixFactory.createRandn(1, featureCount);
    Matrix shiftRowVector = matrixFactory.createRandn(1, featureCount);
    Neurons leftNeurons = new Neurons(featureCount, true);
    Neurons rightNeurons = new Neurons(featureCount, true);
    ScaleAndShiftAxonsConfig config = new ScaleAndShiftAxonsConfig(leftNeurons, rightNeurons, scaleRowVector, shiftRowVector);

    createAxons(leftNeurons, rightNeurons, config);
  }

  @Test
  public void testPushLeftToRightThenRightToLeft() {
	
	  int featureCount = 10;
	    Matrix scaleColumnVector = matrixFactory.createRandn(featureCount, 1);
	    Matrix shiftColumnVector = matrixFactory.createRandn(featureCount, 1);
	    Neurons leftNeurons = new Neurons(featureCount, true);
	    Neurons rightNeurons = new Neurons(featureCount, false);
	    ScaleAndShiftAxonsConfig config = new ScaleAndShiftAxonsConfig(leftNeurons, rightNeurons, scaleColumnVector, shiftColumnVector);

	    ScaleAndShiftAxons<?> axons = createAxons(leftNeurons, rightNeurons, config);

	    Matrix inputMatrix = matrixFactory.createRand(featureCount, 100);
	 
	    Assert.assertEquals(featureCount, inputMatrix.getRows());
	    Assert.assertEquals(100, inputMatrix.getColumns());

	    NeuronsActivation input = new NeuronsActivationImpl(leftNeurons, inputMatrix,
	    		NeuronsActivationFormat.ROWS_SPAN_FEATURE_SET);
	    AxonsContext context = new AxonsContextImpl("someAxons", matrixFactory, false, false);
	    AxonsActivation axonsActivation = axons.pushLeftToRight(input, null, context);
	    NeuronsActivation rightNeuronsActivation = new NeuronsActivationImpl(rightNeurons, axonsActivation.getPostDropoutOutput().getActivations(matrixFactory),
	    		NeuronsActivationFormat.ROWS_SPAN_FEATURE_SET);
	    
	    AxonsActivation finalActivation = axons.pushRightToLeft(rightNeuronsActivation, axonsActivation, context);
	    Matrix finalOutput = finalActivation.getPostDropoutOutput().getActivations(matrixFactory);
	    for (int r = 0; r < inputMatrix.getRows(); r++) {
	        for (int c = 0; c < inputMatrix.getColumns(); c++) {

	          Assert.assertEquals(inputMatrix.get(r, c) , finalOutput.get(r, c),
	              0.000001);
	        }
	      }
  }


  @Test
  public void testPushRightToLeft() {

    int featureCount = 10;
    Matrix scaleColumnVector = matrixFactory.createRandn(featureCount, 1);
    Matrix shiftColumnVector = matrixFactory.createRandn(featureCount, 1);
    Neurons leftNeurons = new Neurons(featureCount, true);
    Neurons rightNeurons = new Neurons(featureCount, false);
    ScaleAndShiftAxonsConfig config = new ScaleAndShiftAxonsConfig(leftNeurons, rightNeurons, scaleColumnVector, shiftColumnVector);

    ScaleAndShiftAxons<?> axons = createAxons(leftNeurons, rightNeurons, config);

    Matrix inputMatrix = matrixFactory.createRand(featureCount, 100);

    Assert.assertEquals(featureCount, inputMatrix.getRows());
    Assert.assertEquals(100, inputMatrix.getColumns());

    NeuronsActivation input = new NeuronsActivationImpl(leftNeurons, inputMatrix,
    		NeuronsActivationFormat.ROWS_SPAN_FEATURE_SET);
    AxonsContext context = new AxonsContextImpl("someAxons", matrixFactory, false,  false);
    AxonsActivation axonsActivation = axons.pushRightToLeft(input, null, context);

    Matrix outputMatrix = axonsActivation.getPostDropoutOutput().getActivations(matrixFactory);

    Assert.assertEquals(NeuronsActivationFeatureOrientation.ROWS_SPAN_FEATURE_SET,
        axonsActivation.getPostDropoutOutput().getFeatureOrientation());


    Assert.assertEquals(inputMatrix.getRows(), outputMatrix.getRows());

    Assert.assertEquals(inputMatrix.getColumns(), outputMatrix.getColumns());

   
    for (int r = 0; r < inputMatrix.getRows(); r++) {
      for (int c = 0; c < inputMatrix.getColumns(); c++) {

        double scale = scaleColumnVector.get(r, 0);
        Assert.assertEquals(inputMatrix.get(r, c) * scale, outputMatrix.get(r, c),
            0.000001);
      }
    }
  }


}
