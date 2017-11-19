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

import org.ml4j.Matrix;

/**
 * The configuration for ScaleAndShiftAxons.
 * 
 * @author Michael Lavelle
 *
 */
public class ScaleAndShiftAxonsConfig extends AxonsConfig {

  /**
   * Default serialization id.
   */
  private static final long serialVersionUID = 1L;
  
  private Matrix scaleRowVector;
  private Matrix shiftRowVector;
  
  /**
   * Configure scale and shift axons.
   * 
   * @param scaleRowVector A row vector consisting of the scalings for each output neuron.
   * @param shiftRowVector A row vector consisting of the shifts for each output neuron.
   */
  public ScaleAndShiftAxonsConfig(Matrix scaleRowVector, Matrix shiftRowVector) {
    super();
    this.scaleRowVector = scaleRowVector;
    this.shiftRowVector = shiftRowVector;
  }

  /**
   * @return A row vector consisting of the scalings for each output neuron.
   */
  public Matrix getScaleRowVector() {
    return scaleRowVector;
  }

  /**
   * @return A row vector consisting of the shifts for each output neuron.
   */
  public Matrix getShiftRowVector() {
    return shiftRowVector;
  }
}