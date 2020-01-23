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
package org.ml4j.nn.components.manytoone.base;

import java.util.Date;
import java.util.List;
import java.util.Optional;

import org.ml4j.nn.components.DirectedComponentsContext;
import org.ml4j.nn.components.NeuralComponentType;
import org.ml4j.nn.components.manytoone.ManyToOneDirectedComponent;
import org.ml4j.nn.components.manytoone.ManyToOneDirectedComponentActivation;
import org.ml4j.nn.components.onetoone.DefaultChainableDirectedComponentAdapter;
import org.ml4j.nn.neurons.NeuronsActivation;
import org.ml4j.nn.neurons.NeuronsActivationFormat;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class ManyToOneDirectedComponentAdapter<A extends ManyToOneDirectedComponentActivation>  
	implements ManyToOneDirectedComponent<A> {

	/**
	 * Default serialization id.
	 */
	private static final long serialVersionUID = 1L;
	
	private static final Logger LOGGER = LoggerFactory.getLogger(ManyToOneDirectedComponentAdapter.class);
	
	private ManyToOneDirectedComponent<A> delegated;
	
	public ManyToOneDirectedComponentAdapter(ManyToOneDirectedComponent<A> delegated) {
		this.delegated = delegated;
	}

	@Override
	public A forwardPropagate(List<NeuronsActivation> input, DirectedComponentsContext context) {
		
		LOGGER.debug(getComponentType().toString());
		
		long startTime = new Date().getTime();
		A activation =  delegated.forwardPropagate(input, context);
		long endTime = new Date().getTime();
		long timeTaken = endTime - startTime;
		DefaultChainableDirectedComponentAdapter.addTime(timeTaken, delegated.getClass().getSimpleName());
		return activation;
	}

	@Override
	public NeuralComponentType<?> getComponentType() {
		return delegated.getComponentType();
	}

	@Override
	public ManyToOneDirectedComponent<A> dup() {
		return new ManyToOneDirectedComponentAdapter<A>(delegated.dup());
	}
	
	@Override
	public boolean isSupported(NeuronsActivationFormat<?> format) {
		return delegated.isSupported(format);
	}

	@Override
	public Optional<NeuronsActivationFormat<?>> optimisedFor() {
		return delegated.optimisedFor();
	}

}
