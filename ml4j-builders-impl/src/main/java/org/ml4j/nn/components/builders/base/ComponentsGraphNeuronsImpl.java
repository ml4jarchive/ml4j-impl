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
package org.ml4j.nn.components.builders.base;

import org.ml4j.nn.components.builders.componentsgraph.ComponentsGraphNeurons;
import org.ml4j.nn.neurons.Neurons;

public class ComponentsGraphNeuronsImpl<N extends Neurons> implements ComponentsGraphNeurons<N> {

	private N currentNeurons;
	private N rightNeurons;
	private boolean hasBiasUnit;
	
	public ComponentsGraphNeuronsImpl(N currentNeurons) {
		this.currentNeurons = currentNeurons;
		if (currentNeurons == null) {
			throw new IllegalArgumentException("Current neurons must not be null!");
		}
	}
	
	public ComponentsGraphNeuronsImpl(N currentNeurons, N rightNeurons) {
		this.currentNeurons = currentNeurons;
		this.rightNeurons = rightNeurons;
		if (currentNeurons == null) {
			throw new IllegalArgumentException("Current neurons must not be null!");
		}
	}
	
	public N getCurrentNeurons() {
		return currentNeurons;
	}
	public void setCurrentNeurons(N currentNeurons) {
		if (currentNeurons == null) {
			throw new IllegalArgumentException("Current neurons must not be null!");
		}
		this.currentNeurons = currentNeurons;
	}
	public N getRightNeurons() {
		return rightNeurons;
	}
	public void setRightNeurons(N rightNeurons) {
		this.rightNeurons = rightNeurons;
	}

	@Override
	public boolean hasBiasUnit() {
		return hasBiasUnit;
	}

	@Override
	public void setHasBiasUnit(boolean hasBiasUnit) {
		this.hasBiasUnit = hasBiasUnit;
	}	
}
