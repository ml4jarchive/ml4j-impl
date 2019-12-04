package org.ml4j.nn.components;

import org.jblas.NativeBlas;
import org.ml4j.Matrix;
import org.ml4j.jblas.JBlasMatrixFactory2;
import org.ml4j.nn.neurons.Neurons3D;

public class Rearranger {

	public static Matrix forwardRearrange(Neurons3D left, Matrix m) {
	
		int channels = left.getDepth();
		int examples = m.getColumns();
		if (examples == 1 && channels == 1) {
			return m;
		} else {
		
		// Swap input channels and examples;
		float[] data = m.getRowByRowArray();
		float[] target = new float[data.length];
		
	
		int sourceStart = 0;
		int sourceIncrement = examples;
		int targetStart = 0;
		int targetIncrement = channels;
		int n = data.length / (examples * channels);
		
		for (int e = 0; e < examples; e++) {
			for (int c = 0; c < channels; c++) {
				NativeBlas.scopy(n, data, sourceStart + e + c * data.length / channels, sourceIncrement, target, targetStart + e * data.length / examples + c, targetIncrement);
			}
		}
	
		return new JBlasMatrixFactory2().createMatrixFromRowsByRowsArray(data.length / channels, channels, target);
		
		}

	}
	
}
