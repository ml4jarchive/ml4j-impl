package org.ml4j.nn.datasets;

public class SoftmaxClassIndexLabelMapper implements FeatureExtractor<Integer> {

	private int totalClasses;

	public SoftmaxClassIndexLabelMapper(int totalClasses) {
		this.totalClasses = totalClasses;
	}

	@Override
	public float[] getFeatures(Integer label) {
		float[] floatArray = new float[totalClasses];
		floatArray[label] = 1;
		return floatArray;
	}

	@Override
	public int getFeatureCount() {
		return totalClasses;
	}

}
