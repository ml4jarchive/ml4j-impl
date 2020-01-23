/*
 * Copyright 2017 the original author or authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
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

public class NeuronsActivationTest {

  private MatrixFactory matrixFactory;

  @Before
  public void setUp() {
    matrixFactory = new JBlasRowMajorMatrixFactory();
  }

  @Test
  public void testConstructorColumnsSpanFeatureSet() {

    Matrix matrix = matrixFactory.createRand(10, 784);
    NeuronsActivation neuronsActivation = new NeuronsActivationImpl(new Neurons(784, false), matrix,
    		NeuronsActivationFormat.COLUMNS_SPAN_FEATURE_SET);
    Matrix activations = neuronsActivation.getActivations(matrixFactory);
    Assert.assertEquals(matrix.getRows(), activations.getRows());
    Assert.assertEquals(matrix.getColumns(), activations.getColumns());
    Assert.assertEquals(NeuronsActivationFeatureOrientation.COLUMNS_SPAN_FEATURE_SET,
        neuronsActivation.getFeatureOrientation());
    Assert.assertEquals(784, neuronsActivation.getFeatureCount());
  }

  @Test
  public void testConstructorRowsSpanFeatureSetWithBiasRequest() {

    Matrix matrix = matrixFactory.createRand(784, 10);

    NeuronsActivation neuronsActivation = new NeuronsActivationImpl(new Neurons(784, false), matrix,
    		NeuronsActivationFormat.ROWS_SPAN_FEATURE_SET);
    Matrix activations = neuronsActivation.getActivations(matrixFactory);
    Assert.assertEquals(matrix.getRows(), activations.getRows());
    Assert.assertEquals(matrix.getColumns(), activations.getColumns());
    Assert.assertEquals(NeuronsActivationFeatureOrientation.ROWS_SPAN_FEATURE_SET,
        neuronsActivation.getFeatureOrientation());
    Assert.assertEquals(784, neuronsActivation.getFeatureCount());


  }


}
