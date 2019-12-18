package org.ml4j.floatarray;

public class DefaultFloatArrayFactory implements FloatArrayFactory {

	/**
	 * Default serialization id
	 */
	private static final long serialVersionUID = 1L;

	@Override
	public float[] createFloatArray(int length) {
		return new float[length];
	}
}
