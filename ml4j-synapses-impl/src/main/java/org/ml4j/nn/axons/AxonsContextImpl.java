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

package org.ml4j.nn.axons;

/**
 * Simple implementation of AxonsContext.
 * 
 * @author Michael Lavelle
 */
import org.ml4j.MatrixFactory;

public class AxonsContextImpl implements AxonsContext {

  /**
   * Default serialization id.
   */
  private static final long serialVersionUID = 1L;
  
  /**
   * The MatrixFactory we configure for this context.
   */
  private MatrixFactory matrixFactory;
  
  private double leftHandDropoutKeepProbability;
  
  private boolean withFreezeOut;
  
  /**
   * Construct a new AxonsContext.
   * 
   * @param matrixFactory The MatrixFactory we configure for this context
   * @param leftHandDropoutKeepProbability The dropout keep-probability on the left hand side of
   *        these Axons.
   */
  public AxonsContextImpl(MatrixFactory matrixFactory, 
      double leftHandDropoutKeepProbability, boolean withFreezeOut) {
    this.matrixFactory = matrixFactory;
    this.leftHandDropoutKeepProbability = leftHandDropoutKeepProbability;
    this.withFreezeOut = withFreezeOut;
  }

  @Override
  public MatrixFactory getMatrixFactory() {
    return matrixFactory;
  }

  @Override
  public double getLeftHandInputDropoutKeepProbability() {
    return leftHandDropoutKeepProbability;
  }

  @Override
  public boolean isWithFreezeOut() {
    return withFreezeOut;
  }
}
