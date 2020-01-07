package org.ml4j.floatarray;

import java.io.Serializable;

public interface FloatArrayFactory extends Serializable {

	float[] createFloatArray(int length);
}
