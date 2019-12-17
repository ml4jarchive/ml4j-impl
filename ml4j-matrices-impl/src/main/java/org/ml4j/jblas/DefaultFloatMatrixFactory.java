package org.ml4j.jblas;

import org.jblas.FloatMatrix;
import org.ml4j.floatmatrix.FloatMatrixFactory;

public class DefaultFloatMatrixFactory implements FloatMatrixFactory {

	  /**
	 * Default serialization id
	 */
	private static final long serialVersionUID = 1L;

	@Override
	public FloatMatrix create(float[][] data) {
		return new FloatMatrix(data);
	}

	@Override
	public FloatMatrix create(int rows, int columns) {
		return new FloatMatrix(rows, columns);
	}

	@Override
	public FloatMatrix create(int rows, int columns, float[] data) {
		return new FloatMatrix(rows, columns, data);
	}
	
}
