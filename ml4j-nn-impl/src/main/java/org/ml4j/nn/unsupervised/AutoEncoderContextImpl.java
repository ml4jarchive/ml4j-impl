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

package org.ml4j.nn.unsupervised;

import org.ml4j.MatrixFactory;
import org.ml4j.nn.layers.DirectedLayerContext;
import org.ml4j.nn.layers.mocks.DirectedLayerContextMock;
import org.ml4j.nn.unsupervised.AutoEncoderContext;

/**
 * Simple default implementation of AutoEncoderContext.
 * 
 * @author Michael Lavelle
 * 
 */
public class AutoEncoderContextImpl implements AutoEncoderContext {

  /**
   * Default serialization id.
   */
  private static final long serialVersionUID = 1L;

  /**
   * The MatrixFactory we configure for this context.
   */
  private MatrixFactory matrixFactory;

  private int startLayerIndex;
  
  private Integer endLayerIndex;
 
  /**
   * Construct a default AutoEncoderContext.
   * 
   * @param matrixFactory The MatrixFactory we configure for this context
   */
  public AutoEncoderContextImpl(MatrixFactory matrixFactory, 
      int startLayerIndex, Integer endLayerIndex) {
    this.matrixFactory = matrixFactory;
    this.startLayerIndex = startLayerIndex;
    this.endLayerIndex = endLayerIndex;
    if (endLayerIndex != null && startLayerIndex > endLayerIndex) {
      throw new IllegalArgumentException("Start layer index cannot be greater "
          + "than end layer index");
    }
  }

  @Override
  public MatrixFactory getMatrixFactory() {
    return matrixFactory;
  }

  @Override
  public DirectedLayerContext createLayerContext(int layerIndex) {
    return new DirectedLayerContextMock(matrixFactory);
  }

  @Override
  public int getStartLayerIndex() {
    return startLayerIndex;
  }

  @Override
  public Integer getEndLayerIndex() {
    return endLayerIndex;
  }
}
