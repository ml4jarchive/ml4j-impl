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

package org.ml4j.nn.costfunctions;

import org.ml4j.Matrix;

/**
 * Cross entropy cost function.
 * 
 * @author Michael Lavelle
 *
 */
public class CrossEntropyCostFunction implements CostFunction {

  @Override
  public double getTotalCost(Matrix desiredOutputs, Matrix actualOutputs) {
    Matrix jpart = (desiredOutputs.mul(-1).mul(actualOutputs.log())
        .sub(desiredOutputs.mul(-1).add(1).mul(actualOutputs.mul(-1).add(1).log()))).rowSums();
    return jpart.sum();
  }

  @Override
  public double getAverageCost(Matrix desiredOutputs, Matrix actualOutputs) {
    int numberOfExamples = desiredOutputs.getRows();
    return getTotalCost(desiredOutputs, actualOutputs) / numberOfExamples;
  }
}
