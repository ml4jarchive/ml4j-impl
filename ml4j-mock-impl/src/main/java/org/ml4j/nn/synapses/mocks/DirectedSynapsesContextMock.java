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

package org.ml4j.nn.synapses.mocks;

import org.ml4j.MatrixFactory;
import org.ml4j.nn.axons.AxonsContext;
import org.ml4j.nn.axons.mocks.AxonsContextMock;
import org.ml4j.nn.synapses.DirectedSynapsesContext;

/**
 * Simple mock implementation of DirectedSynapsesContext.
 * 
 * @author Michael Lavelle
 * 
 */
public class DirectedSynapsesContextMock implements DirectedSynapsesContext {

  /**
   * Default serialization id.
   */
  private static final long serialVersionUID = 1L;
  
  /**
   * The MatrixFactory we configure for this context.
   */
  private MatrixFactory matrixFactory;
  
  /**
   * Construct a new mock DirectedSynapsesContext.
   * 
   * @param matrixFactory The MatrixFactory we configure for this context
   */
  public DirectedSynapsesContextMock(MatrixFactory matrixFactory) {
    this.matrixFactory = matrixFactory;
  }
 
  @Override
  public MatrixFactory getMatrixFactory() {
    return matrixFactory;
  }

  @Override
  public AxonsContext createAxonsContext() {
    return new AxonsContextMock(matrixFactory);
  }
}
