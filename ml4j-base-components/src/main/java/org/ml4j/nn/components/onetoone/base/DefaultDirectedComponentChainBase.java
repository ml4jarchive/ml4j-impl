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
package org.ml4j.nn.components.onetoone.base;



import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;

import org.ml4j.nn.components.NeuralComponentBaseType;
import org.ml4j.nn.components.NeuralComponentType;
import org.ml4j.nn.components.NeuralComponentVisitor;
import org.ml4j.nn.components.onetone.DefaultChainableDirectedComponent;
import org.ml4j.nn.components.onetone.DefaultChainableDirectedComponentActivation;
import org.ml4j.nn.components.onetone.DefaultDirectedComponentChain;
import org.ml4j.nn.components.onetone.DefaultDirectedComponentChainActivation;

/**
 * Default base class for implementations of DefaultDirectedComponentChain.
 * 
 * Encapsulates a sequential chain of DefaultChainableDirectedComponents
 * 
 * @author Michael Lavelle
 */
public abstract class DefaultDirectedComponentChainBase 
		extends DefaultDirectedComponentChainBaseParent<DefaultChainableDirectedComponent<?, ?>, DefaultChainableDirectedComponentActivation, DefaultDirectedComponentChainActivation> 
		implements DefaultDirectedComponentChain {

	/**
	 * Default serialization id.
	 */
	private static final long serialVersionUID = 1L;

	public DefaultDirectedComponentChainBase(List<DefaultChainableDirectedComponent<?, ?>> sequentialComponents) {
		super(sequentialComponents);
	}
	
	@Override
	public NeuralComponentType getComponentType() {
		return NeuralComponentType.createSubType(NeuralComponentType.getBaseType(NeuralComponentBaseType.COMPONENT_CHAIN), DefaultDirectedComponentChain.class.getName());
	}
	
	@Override
	public String getName() {
		return getComponentType().getId();
	}
	
	@Override
	public String accept(NeuralComponentVisitor<DefaultChainableDirectedComponent<?, ?>> visitor) {
		return visitor.visitComponent(this);
	}
	
	@Override
	public Set<DefaultChainableDirectedComponent<?, ?>> flatten() {
		Set<DefaultChainableDirectedComponent<?, ?>> allComponentsIncludingThis = new HashSet<>(Arrays.asList(this));
		allComponentsIncludingThis.addAll(this.sequentialComponents.stream().flatMap(c -> c.flatten().stream()).collect(Collectors.toSet()));
		return allComponentsIncludingThis;
	}

}
