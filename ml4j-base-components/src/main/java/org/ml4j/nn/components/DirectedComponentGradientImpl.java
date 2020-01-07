/*
 * Copyright 2019 the original author or authors.
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
package org.ml4j.nn.components;

import java.util.ArrayList;
import java.util.List;
import java.util.function.Supplier;

import org.ml4j.nn.axons.AxonsGradient;

public class DirectedComponentGradientImpl<O> implements DirectedComponentGradient<O> {

	private O output;
	private List<Supplier<AxonsGradient>> totalTrainableAxonsGradients;
	//private List<AxonsGradient> averageTrainableAxonsGradients;

	
	public DirectedComponentGradientImpl(List<Supplier<AxonsGradient>> totalTrainableAxonsGradients, Supplier<AxonsGradient> axonsGradient, O output) {
		this.totalTrainableAxonsGradients = new ArrayList<>();
		this.totalTrainableAxonsGradients.addAll(totalTrainableAxonsGradients);
		this.totalTrainableAxonsGradients.add(axonsGradient);
		this.output = output;
	}
	
	public DirectedComponentGradientImpl(List<Supplier<AxonsGradient>> totalTrainableAxonsGradients, O output) {
		this.totalTrainableAxonsGradients = new ArrayList<>();
		this.totalTrainableAxonsGradients.addAll(totalTrainableAxonsGradients);
		this.output = output;
	}


	public DirectedComponentGradientImpl(DirectedComponentGradient<?> previousGradient, O output) {
		this.totalTrainableAxonsGradients = new ArrayList<>();
		this.totalTrainableAxonsGradients.addAll(previousGradient.getTotalTrainableAxonsGradients());
		this.output = output;
	}


	public DirectedComponentGradientImpl(O output) {
		this.totalTrainableAxonsGradients = new ArrayList<>();
		this.output = output;
	}



	@Override
	public O getOutput() {
		return output;
	}

	@Override
	public void addTotalTrainableAxonsGradient(Supplier<AxonsGradient> axonsGradient) {
		totalTrainableAxonsGradients.add(axonsGradient);
	}

	@Override
	public List<Supplier<AxonsGradient>> getTotalTrainableAxonsGradients() {
		return totalTrainableAxonsGradients;
	}

	@Override
	public List<Supplier<AxonsGradient>> getAverageTrainableAxonsGradients() {
		throw new UnsupportedOperationException("Not yet supported");
	}
}
