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
package org.ml4j.nn.components.axons.base;

import java.util.Date;
import java.util.List;
import java.util.function.Supplier;
import java.util.stream.Collectors;

import org.ml4j.nn.axons.AxonsGradient;
import org.ml4j.nn.components.DirectedComponentGradient;
import org.ml4j.nn.components.DirectedComponentGradientImpl;
import org.ml4j.nn.components.axons.DirectedAxonsComponent;
import org.ml4j.nn.components.axons.DirectedAxonsComponentActivation;
import org.ml4j.nn.components.onetone.DefaultChainableDirectedComponentActivation;
import org.ml4j.nn.components.onetoone.DefaultChainableDirectedComponentAdapter;
import org.ml4j.nn.neurons.NeuronsActivation;

public class DirectedAxonsComponentActivationAdapter implements DirectedAxonsComponentActivation {

	private DirectedAxonsComponentActivation delegated;
	private String name;
	
	public DirectedAxonsComponentActivationAdapter(DirectedAxonsComponentActivation delegated, String name) {
		this.delegated = delegated;
		this.name = name;
	}
	
	@Override
	public List<? extends DefaultChainableDirectedComponentActivation> decompose() {
		return delegated.decompose();
	}

	@Override
	public DirectedComponentGradient<NeuronsActivation> backPropagate(
			DirectedComponentGradient<NeuronsActivation> gradient) {
		long startTime = new Date().getTime();
		DirectedComponentGradient<NeuronsActivation> grad =  delegated.backPropagate(gradient);
		long endTime = new Date().getTime();
		long timeTaken = endTime - startTime;
		DefaultChainableDirectedComponentAdapter.addTime(timeTaken, "bp:" + name);
		return new DirectedComponentGradientImpl<>(grad.getTotalTrainableAxonsGradients().stream()
				.map(s -> decorateGradientSupplier(s)).collect(Collectors.toList()), grad.getOutput());
	}
	
	private Supplier<AxonsGradient> decorateGradientSupplier(Supplier<AxonsGradient> gradientSupplier) {
		return gradientSupplier == null ? null : new GradientSupplierAdapter(gradientSupplier); 
	}
	
	private class GradientSupplierAdapter implements Supplier<AxonsGradient> {

		private Supplier<AxonsGradient> gradientSupplier;
		
		public GradientSupplierAdapter(Supplier<AxonsGradient> gradientSupplier) {
			this.gradientSupplier = gradientSupplier;
		}
		
		@Override
		public AxonsGradient get() {
			long startTime = new Date().getTime();
			AxonsGradient gradient = gradientSupplier.get();
			long endTime = new Date().getTime();
			long timeTaken = endTime - startTime;
			DefaultChainableDirectedComponentAdapter.addTime(timeTaken, "grad:" + name);
			return gradient;
		}
		
	}

	@Override
	public NeuronsActivation getOutput() {
		return delegated.getOutput();
	}

	@Override
	public double getAverageRegularisationCost() {
		return delegated.getAverageRegularisationCost();
	}

	@Override
	public DirectedAxonsComponent<?, ?, ?> getAxonsComponent() {
		return delegated.getAxonsComponent();
	}

	@Override
	public float getTotalRegularisationCost() {
		return delegated.getTotalRegularisationCost();
	}

}
