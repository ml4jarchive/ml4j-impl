package org.ml4j.nn.axons.base;

import org.ml4j.nn.axons.Axons;
import org.ml4j.nn.neurons.Neurons;

public abstract class AxonsBase<L extends Neurons, R extends Neurons, A extends Axons<L, R, A>> implements Axons<L, R, A>{

	/**
	 * Default serialization id.
	 */
	private static final long serialVersionUID = 1L;
	
	protected L leftNeurons;
	protected R rightNeurons;
	
	
	public AxonsBase(L leftNeurons, R rightNeurons) {
		this.leftNeurons = leftNeurons;
		this.rightNeurons = rightNeurons;
	}

	@Override
	public L getLeftNeurons() {
		return leftNeurons;
	}

	@Override
	public R getRightNeurons() {
		return rightNeurons;
	}
}
