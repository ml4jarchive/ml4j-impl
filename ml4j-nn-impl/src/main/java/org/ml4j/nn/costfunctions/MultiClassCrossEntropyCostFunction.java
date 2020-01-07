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

package org.ml4j.nn.costfunctions;

import org.ml4j.EditableMatrix;
import org.ml4j.InterrimMatrix;
import org.ml4j.Matrix;

/**
 * Multi class cross entropy cost function.
 * 
 * @author Michael Lavelle
 *
 */
public class MultiClassCrossEntropyCostFunction implements CostFunction {

  @Override
  public float getTotalCost(Matrix desiredOutputs, Matrix actualOutputs) {
   
	    if (actualOutputs.getColumns() != desiredOutputs.getColumns()) {
	    	 throw new IllegalArgumentException("Mismatched column count between desired and actual outputs");
	    }
	    if (actualOutputs.getRows() != desiredOutputs.getRows()) {
	   	 throw new IllegalArgumentException("Mismatched row count between desired and actual outputs");
	   }
	  
		try (InterrimMatrix limitLog = limitLog(actualOutputs).asInterrimMatrix()) {
			try (InterrimMatrix negativeOfDesiredOutputs = desiredOutputs.mul(-1).asInterrimMatrix()) {
				try (InterrimMatrix jpart = (negativeOfDesiredOutputs.asEditableMatrix().muli(limitLog)).rowSums().asInterrimMatrix()) {
					return jpart.sum();
				}
			}

		}
   
  }

  private double limit(float value) {
	  // Removed 4 0's
    value = Math.min(value, 1 - 0.00000000001f);
    value = Math.max(value, 0.00000000001f);
    return value;
  }

  private Matrix limitLog(Matrix matrix) {
    EditableMatrix dupMatrix = matrix.dup().asEditableMatrix();
    for (int r = 0; r < dupMatrix.getRows(); r++) {
    	for (int c = 0; c < dupMatrix.getColumns(); c++) {
    	      dupMatrix.put(r, c, (float) Math.log(limit(dupMatrix.get(r, c))));
    	}
    }
    return dupMatrix;
  }

  @Override
  public float getAverageCost(Matrix desiredOutputs, Matrix actualOutputs) {
    return getTotalCost(desiredOutputs, actualOutputs)/desiredOutputs.getRows();
  }
}
