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

package org.ml4j;

import org.ml4j.Matrix;
import org.ml4j.MatrixFactory;
import org.ml4j.mocks.MatrixFactoryMock;

/**
 * The default ml4j MatrixFactory.
 * 
 * @author Michael Lavelle
 */
public class MatrixFactoryImpl implements MatrixFactory {

  private MatrixFactory delegatedFactory;

  public MatrixFactoryImpl() {
    this.delegatedFactory = new MatrixFactoryMock();
  }


  @Override
  public Matrix createOnes(int rows, int columns) {
    return delegatedFactory.createOnes(rows, columns);
  }

  @Override
  public Matrix createMatrix(double[][] data) {
    return delegatedFactory.createMatrix(data);
  }

  @Override
  public Matrix createZeros(int rows, int columns) {
    throw new UnsupportedOperationException("Not implemented yet");
  }

  @Override
  public Matrix createRandn(int rows, int columns) {
    throw new UnsupportedOperationException("Not implemented yet");
  }
}
