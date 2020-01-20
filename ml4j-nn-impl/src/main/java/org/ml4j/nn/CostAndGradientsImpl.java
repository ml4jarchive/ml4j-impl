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

package org.ml4j.nn;

import java.util.ArrayList;
import java.util.List;

import org.ml4j.nn.axons.AxonsGradient;
import org.ml4j.nn.axons.AxonsGradientImpl;

/**
 * Encapsulates the total and average costs and gradients associated with a
 * forward propagation through a FeedForwardNeuralNetwork.
 * 
 * @author Michael Lavelle
 */
public class CostAndGradientsImpl implements CostAndGradients {

	private float totalCost;

	private List<AxonsGradient> totalTrainableAxonsGradients;

	private int numberOfTrainingExamples;

	/**
	 * @param totalCost                    The totalCost.
	 * @param totalTrainableAxonsGradients The totalGradients.
	 * @param numberOfTrainingExamples     The number of training examples.
	 */
	public CostAndGradientsImpl(float totalCost, List<AxonsGradient> totalTrainableAxonsGradients,
			int numberOfTrainingExamples) {
		super();
		this.totalCost = totalCost;
		this.totalTrainableAxonsGradients = totalTrainableAxonsGradients;
		this.numberOfTrainingExamples = numberOfTrainingExamples;
	}

	public float getTotalCost() {
		return totalCost;
	}

	public float getAverageCost() {
		return getTotalCost() / numberOfTrainingExamples;
	}

	public List<AxonsGradient> getTotalTrainableAxonsGradients() {
		return totalTrainableAxonsGradients;
	}

	/**
	 * @return The average gradients.
	 */
	public List<AxonsGradient> getAverageTrainableAxonsGradients() {

		List<AxonsGradient> averages = new ArrayList<>();
		for (AxonsGradient total : getTotalTrainableAxonsGradients()) {
			averages.add(
					new AxonsGradientImpl(total.getAxons(), total.getWeightsGradient().div(numberOfTrainingExamples),
							total.getLeftToRightBiasGradient() == null ? null
									: total.getLeftToRightBiasGradient().div(numberOfTrainingExamples)));
		}
		return averages;
	}

	@Override
	public void close() {
		for (AxonsGradient axonsGradient : totalTrainableAxonsGradients) {
			axonsGradient.getWeightsGradient().close();
			if (axonsGradient.getLeftToRightBiasGradient() != null) {
				axonsGradient.getLeftToRightBiasGradient().close();
			}
			if (axonsGradient.getRightToLeftBiasGradient() != null) {
				axonsGradient.getRightToLeftBiasGradient().close();
			}

		}
	}
}
