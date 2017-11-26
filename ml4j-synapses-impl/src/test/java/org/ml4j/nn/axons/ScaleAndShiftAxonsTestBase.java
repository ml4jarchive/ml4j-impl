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
import org.ml4j.jblas.JBlasMatrixFactory;
import org.ml4j.nn.neurons.Neurons;
import org.ml4j.nn.neurons.NeuronsActivation;
import org.ml4j.nn.neurons.NeuronsActivationFeatureOrientation;

public abstract class ScaleAndShiftAxonsTestBase {

  protected MatrixFactory matrixFactory;

  @Before
  public void setUp() {
    matrixFactory = new JBlasMatrixFactory();
  }

  protected abstract ScaleAndShiftAxons createAxons(Neurons leftNeurons, Neurons rightNeurons,
      ScaleAndShiftAxonsConfig config);

  @Test(expected=IllegalArgumentException.class)
  public void testPushLeftToRightWithNoLeftHandBias() {

    int featureCount = 10;
    Matrix scaleRowVector = matrixFactory.createRandn(1, featureCount);
    Matrix shiftRowVector = matrixFactory.createRandn(1, featureCount);
    ScaleAndShiftAxonsConfig config = new ScaleAndShiftAxonsConfig(scaleRowVector, shiftRowVector);
    Neurons leftNeurons = new Neurons(featureCount, false);
    Neurons rightNeurons = new Neurons(featureCount, false);

    createAxons(leftNeurons, rightNeurons, config);
  }
  
  @Test
  public void testPushLeftToRightWithLeftHandBias() {

    int featureCount = 10;
    Matrix scaleRowVector = matrixFactory.createRandn(1, featureCount);
    Matrix shiftRowVector = matrixFactory.createRandn(1, featureCount);
    ScaleAndShiftAxonsConfig config = new ScaleAndShiftAxonsConfig(scaleRowVector, shiftRowVector);
    Neurons leftNeurons = new Neurons(featureCount, true);
    Neurons rightNeurons = new Neurons(featureCount, false);

    ScaleAndShiftAxons axons = createAxons(leftNeurons, rightNeurons, config);

    Matrix inputMatrix = matrixFactory.createRand(100, featureCount);
    inputMatrix = matrixFactory.createOnes(inputMatrix.getRows(), 1)
        .appendHorizontally(inputMatrix);

    Assert.assertEquals(featureCount + 1, inputMatrix.getColumns());
    Assert.assertEquals(100, inputMatrix.getRows());

    NeuronsActivation input = new NeuronsActivation(inputMatrix, true,
        NeuronsActivationFeatureOrientation.COLUMNS_SPAN_FEATURE_SET);
    AxonsContext context = new AxonsContextImpl(matrixFactory, 1);
    AxonsActivation axonsActivation = axons.pushLeftToRight(input, null, context);

    Matrix outputMatrix = axonsActivation.getOutput().getActivations();

    Assert.assertEquals(inputMatrix.getRows(), outputMatrix.getRows());

    Assert.assertEquals(inputMatrix.getColumns() - 1, outputMatrix.getColumns());

    for (int r = 0; r < outputMatrix.getRows(); r++) {
      for (int c = 0; c < outputMatrix.getColumns(); c++) {

        double scale = scaleRowVector.get(0, c);
        double shift = shiftRowVector.get(0, c);

        Assert.assertEquals(inputMatrix.get(r, c + 1) * scale + shift, outputMatrix.get(r, c),
            0.00000000001);
      }
    }
  }
  
  @Test(expected=IllegalArgumentException.class)
  public void testPushLeftToRightWithLeftHandAndRightHandBias() {

    int featureCount = 10;
    Matrix scaleRowVector = matrixFactory.createRandn(1, featureCount);
    Matrix shiftRowVector = matrixFactory.createRandn(1, featureCount);
    ScaleAndShiftAxonsConfig config = new ScaleAndShiftAxonsConfig(scaleRowVector, shiftRowVector);
    Neurons leftNeurons = new Neurons(featureCount, true);
    Neurons rightNeurons = new Neurons(featureCount, true);

    createAxons(leftNeurons, rightNeurons, config);
  }

 

  @Test
  public void testPushRightToLeft() {

    int featureCount = 10;
    Matrix scaleRowVector = matrixFactory.createRandn(1, featureCount);
    Matrix shiftRowVector = matrixFactory.createRandn(1, featureCount);
    ScaleAndShiftAxonsConfig config = new ScaleAndShiftAxonsConfig(scaleRowVector, shiftRowVector);
    Neurons leftNeurons = new Neurons(featureCount, true);
    Neurons rightNeurons = new Neurons(featureCount, false);

    ScaleAndShiftAxons axons = createAxons(leftNeurons, rightNeurons, config);

    Matrix inputMatrix = matrixFactory.createRand(featureCount, 100);

    Assert.assertEquals(featureCount, inputMatrix.getRows());
    Assert.assertEquals(100, inputMatrix.getColumns());

    NeuronsActivation input = new NeuronsActivation(inputMatrix, false,
        NeuronsActivationFeatureOrientation.ROWS_SPAN_FEATURE_SET);
    AxonsContext context = new AxonsContextImpl(matrixFactory, 1);
    AxonsActivation axonsActivation = axons.pushRightToLeft(input, null, context);

    Matrix outputMatrix = axonsActivation.getOutput().getActivations();

    Assert.assertEquals(NeuronsActivationFeatureOrientation.ROWS_SPAN_FEATURE_SET,
        axonsActivation.getOutput().getFeatureOrientation());


    Assert.assertEquals(inputMatrix.getRows() + 1, outputMatrix.getRows());

    Assert.assertEquals(inputMatrix.getColumns(), outputMatrix.getColumns());

    // Output row 0
    for (int c = 0; c < inputMatrix.getColumns(); c++)  {
      double val = 1;
      Assert.assertEquals(val, outputMatrix.get(0, c), 0.00000000001);
    }
    for (int r = 1; r < inputMatrix.getRows(); r++) {
      for (int c = 0; c < inputMatrix.getColumns(); c++) {

        double scale = scaleRowVector.get(0, r - 1);
        Assert.assertEquals(inputMatrix.get(r - 1, c) * scale, outputMatrix.get(r, c),
            0.00000000001);
      }
    }
  }


}
