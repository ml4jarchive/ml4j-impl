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
