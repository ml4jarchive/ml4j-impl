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

package org.ml4j.nn.unsupervised;

import org.ml4j.MatrixFactory;
import org.ml4j.nn.layers.DirectedLayerContext;

/**
 * Default implementation of AutoEncoderContext.
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
   * Construct a new mock AutoEncoderContext.
   * 
   * @param matrixFactory The MatrixFactory we configure for this context
   */
  public AutoEncoderContextImpl(MatrixFactory matrixFactory, int startLayerIndex,
      Integer endLayerIndex) {
    throw new UnsupportedOperationException("Not implemented yet");
  }

  @Override
  public MatrixFactory getMatrixFactory() {
    throw new UnsupportedOperationException("Not implemented yet");
  }

  @Override
  public DirectedLayerContext createLayerContext(int layerIndex) {
    throw new UnsupportedOperationException("Not implemented yet");
  }

  @Override
  public int getStartLayerIndex() {
    throw new UnsupportedOperationException("Not implemented yet");
  }

  @Override
  public Integer getEndLayerIndex() {
    throw new UnsupportedOperationException("Not implemented yet");
  }
}
