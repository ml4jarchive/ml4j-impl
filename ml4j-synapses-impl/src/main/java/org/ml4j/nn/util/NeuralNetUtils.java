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

package org.ml4j.nn.util;

import org.ml4j.Matrix;

/**
 * Neural Network Utils.
 * 
 * @author Michael Lavelle
 */
public class NeuralNetUtils {

  /**
   * Returns a matrix that has the first derivative of the sigmoid function applied to each element
   * of given input matrix. http://en.wikipedia.org/wiki/Sigmoid_function
   */
  public static Matrix sigmoidGradient(Matrix input) {
    Matrix result = null;
    result = input;
    Matrix sig = result.sigmoid();
    result = sig.subi(sig.mul(sig));
    return result;
  }

  /**
   * Returns a matrix that has the softmax function applied to each element of given input matrix.
   */
  public static Matrix softmax(Matrix x1) {
    Matrix exp = x1.expi();
    Matrix sums = exp.rowSums();
    return exp.diviColumnVector(sums);
  }
}
