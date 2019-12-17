package org.ml4j.nn.axons;

import org.ml4j.Matrix;

public class AxonsDropoutMaskImpl implements AxonsDropoutMask {
	
	private Matrix dropoutMask;
	private AxonsDropoutMaskType type;

	public AxonsDropoutMaskImpl(Matrix dropoutMask, AxonsDropoutMaskType type) {
		this.dropoutMask = dropoutMask;
		this.type = type;
	}
	
	@Override
	public Matrix getDropoutMask() {
		return dropoutMask;
	}

	@Override
	public AxonsDropoutMaskType getType() {
		return type;
	}

}
