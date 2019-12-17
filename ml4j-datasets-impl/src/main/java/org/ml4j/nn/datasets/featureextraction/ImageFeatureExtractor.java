package org.ml4j.nn.datasets.featureextraction;

import org.ml4j.images.Image;
import org.ml4j.nn.datasets.FeatureExtractor;
import org.ml4j.nn.datasets.exceptions.FeatureExtractionException;

public class ImageFeatureExtractor implements FeatureExtractor<Image> {

	private int featureCount;

	public ImageFeatureExtractor(int featureCount) {
		this.featureCount = featureCount;
	}

	@Override
	public float[] getFeatures(Image data) throws FeatureExtractionException {
		float[] features = data.getData();
		if (features == null) {
			throw new FeatureExtractionException("Image data was null");
		}
		return features;
	}

	@Override
	public int getFeatureCount() {
		return featureCount;
	}

}
