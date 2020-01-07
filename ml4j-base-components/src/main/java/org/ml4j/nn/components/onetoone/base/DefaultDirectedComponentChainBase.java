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



import java.util.List;

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

}
