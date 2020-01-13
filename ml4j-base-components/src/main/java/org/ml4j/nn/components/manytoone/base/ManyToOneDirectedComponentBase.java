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

import org.ml4j.nn.components.NeuralComponentBaseType;
import org.ml4j.nn.components.NeuralComponentType;
import org.ml4j.nn.components.manytoone.ManyToOneDirectedComponent;
import org.ml4j.nn.components.manytoone.ManyToOneDirectedComponentActivation;
import org.ml4j.nn.components.manytoone.PathCombinationStrategy;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public abstract class ManyToOneDirectedComponentBase<A extends ManyToOneDirectedComponentActivation> implements ManyToOneDirectedComponent<A> {

	@SuppressWarnings("unused")
	private static final Logger LOGGER = LoggerFactory.getLogger(ManyToOneDirectedComponentBase.class);
	
	/**
	 * Serialization id.
	 */
	private static final long serialVersionUID = -7049642040068320620L;
	
	protected PathCombinationStrategy pathCombinationStrategy;
	
	public ManyToOneDirectedComponentBase(PathCombinationStrategy pathCombinationStrategy) {
		this.pathCombinationStrategy = pathCombinationStrategy;
	}

	@Override
	public NeuralComponentType<?> getComponentType() {
		return NeuralComponentType.getBaseType(NeuralComponentBaseType.MANY_TO_ONE);
	}

}
