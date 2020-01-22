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

import org.ml4j.nn.axons.Axons;
import org.ml4j.nn.axons.AxonsContext;
import org.ml4j.nn.components.axons.DirectedAxonsComponent;
import org.ml4j.nn.components.axons.DirectedAxonsComponentActivation;
import org.ml4j.nn.components.onetoone.DefaultChainableDirectedComponentAdapter;
import org.ml4j.nn.neurons.Neurons;
import org.ml4j.nn.neurons.NeuronsActivation;

public class DirectedAxonsComponentAdapter<L extends Neurons, R extends Neurons> extends DefaultChainableDirectedComponentAdapter<DirectedAxonsComponentActivation, AxonsContext> 
	implements DirectedAxonsComponent<L, R, Axons<L, R, ?>> {

	/**
	 * Default serialization id.
	 */
	private static final long serialVersionUID = 1L;

	public DirectedAxonsComponentAdapter(
			DirectedAxonsComponent<L, R, ?> delegated) {
		super(delegated, delegated.getClass().getSimpleName() + ":" + delegated.getAxons().getClass().getSimpleName());
	}

	@SuppressWarnings("unchecked")
	@Override
	public Axons<L, R, ?> getAxons() {
		return (Axons<L, R, ?>)((DirectedAxonsComponent<L, R, ?>)delegated).getAxons();
	}

	@SuppressWarnings("unchecked")
	@Override
	public DirectedAxonsComponentAdapter<L, R> dup() {
		return new DirectedAxonsComponentAdapter<L, R>((DirectedAxonsComponent<L, R, ?>) delegated.dup());
	}
}
