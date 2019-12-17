package org.ml4j.nn.datasets.featureextraction;

import java.util.function.Supplier;

import org.ml4j.images.Image;
import org.ml4j.nn.datasets.FeatureExtractor;
import org.ml4j.nn.datasets.exceptions.FeatureExtractionException;

public class ImageSupplierFeatureExtractor implements FeatureExtractor<Supplier<Image>> {

	private int featureCount;

	public ImageSupplierFeatureExtractor(int featureCount) {
		this.featureCount = featureCount;
	}

	@Override
	public float[] getFeatures(Supplier<Image> data) throws FeatureExtractionException {
		try {
			Image image = data.get();
			if (image == null) {
				throw new FeatureExtractionException("Image returned by image supplier was null");
			} else {
				return image.getData();
			}
		} catch (Exception e) {
			throw new FeatureExtractionException("Unable to obtain image from image supplier", e);
		}
	}

	@Override
	public int getFeatureCount() {
		return featureCount;
	}

}
