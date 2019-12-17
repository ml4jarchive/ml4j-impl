package org.ml4j.nn.datasets;

import java.util.function.Function;

public class SoftmaxEnumIndexLabelMapper<E extends Enum<E>> implements FeatureExtractor<E> {

	private int totalClasses;
	private Function<E, Integer> enumIndexMapper;

	public SoftmaxEnumIndexLabelMapper(Class<E> enumClass, Function<E, Integer> enumIndexMapper) {
		this.totalClasses = enumClass.getEnumConstants().length;
		this.enumIndexMapper = enumIndexMapper;
	}

	@Override
	public float[] getFeatures(E value) {
		float[] floatArray = new float[totalClasses];
		floatArray[enumIndexMapper.apply(value)] = 1;
		return floatArray;
	}

	@Override
	public int getFeatureCount() {
		return totalClasses;
	}

}
