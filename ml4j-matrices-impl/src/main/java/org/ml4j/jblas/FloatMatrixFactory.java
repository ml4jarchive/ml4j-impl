package org.ml4j.jblas;

import java.io.Serializable;

import org.jblas.FloatMatrix;

public interface FloatMatrixFactory extends Serializable {

	FloatMatrix create(float[][] data);

	FloatMatrix create(int rows, int columns);
	
	FloatMatrix create(int rows, int columns, float[] data);

}
