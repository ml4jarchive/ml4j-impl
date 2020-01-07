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
package org.ml4j.nn.components.builders.axons;

import java.util.function.Consumer;
import java.util.function.Supplier;

import org.ml4j.nn.axons.AxonsContext;
import org.ml4j.nn.components.DirectedComponentsContext;
import org.ml4j.nn.neurons.Neurons;

public abstract class UncompletedAxonsBuilderImpl<N extends Neurons, C> implements UncompletedAxonsBuilder<N, C> {

	protected Supplier<C> previousBuilderSupplier;
	protected N leftNeurons;
	protected DirectedComponentsContext directedComponentsContext;
	protected Consumer<AxonsContext> axonsContextConfigurer;
	
	public UncompletedAxonsBuilderImpl(Supplier<C> previousBuilderSupplier, N leftNeurons) {
		this.previousBuilderSupplier = previousBuilderSupplier;
		this.leftNeurons = leftNeurons;
	}
	
	public N getLeftNeurons() {
		return leftNeurons;
	}

	public DirectedComponentsContext getDirectedComponentsContext() {
		return directedComponentsContext;
	}

	public Consumer<AxonsContext> getAxonsContextConfigurer() {
		return axonsContextConfigurer;
	}
	
	
	
}
