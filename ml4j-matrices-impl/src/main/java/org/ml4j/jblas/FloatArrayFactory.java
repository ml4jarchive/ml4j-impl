package org.ml4j.jblas;

import java.io.Serializable;

public interface FloatArrayFactory extends Serializable {

	float[] createFloatArray(int length);
}
