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

import org.ml4j.InterrimMatrix;
import org.ml4j.Matrix;

/**
 * Cross entropy cost function.
 * 
 * @author Michael Lavelle
 *
 */
public class CrossEntropyCostFunction implements CostFunction {

	@Override
	public float getTotalCost(Matrix desiredOutputs, Matrix actualOutputs) {

		try (InterrimMatrix logOfActualOutputs = actualOutputs.log().asInterrimMatrix()) {
			try (InterrimMatrix negativeOfDesiredOutputs = desiredOutputs.mul(-1).asInterrimMatrix()) {
				try (InterrimMatrix negativeOfActualOutputsAddOne = actualOutputs.mul(-1).asEditableMatrix().addi(1)
						.asInterrimMatrix()) {
					try (InterrimMatrix first = negativeOfDesiredOutputs.mul(logOfActualOutputs).asInterrimMatrix()) {
						try (InterrimMatrix second = negativeOfActualOutputsAddOne.logi().asInterrimMatrix()) {
							try (InterrimMatrix third = negativeOfDesiredOutputs.asEditableMatrix().addi(1).muli(second)
									.asInterrimMatrix()) {
								try (InterrimMatrix jpart = first.asEditableMatrix().subi(third).asInterrimMatrix()) {
									try (InterrimMatrix jpartRowSums = jpart.rowSums().asInterrimMatrix()) {
										return jpartRowSums.sum();
									}
								}
							}
						}
					}
				}
			}
		}
	}

	@Override
	public float getAverageCost(Matrix desiredOutputs, Matrix actualOutputs) {
		int numberOfExamples = desiredOutputs.getRows();
		return getTotalCost(desiredOutputs, actualOutputs) / numberOfExamples;
	}
}
