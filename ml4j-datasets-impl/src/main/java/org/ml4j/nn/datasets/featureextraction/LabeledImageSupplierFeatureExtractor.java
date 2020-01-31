/*
 * Copyright 2019 the original author or authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
 * in compliance with the License. You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software distributed under the License
 * is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
 * or implied. See the License for the specific language governing permissions and limitations under
 * the License.
 */
package org.ml4j.nn.datasets.featureextraction;

import java.util.function.Supplier;

import org.ml4j.images.Image;
import org.ml4j.nn.datasets.FeatureExtractor;
import org.ml4j.nn.datasets.LabeledData;
import org.ml4j.nn.datasets.exceptions.FeatureExtractionException;

public class LabeledImageSupplierFeatureExtractor<L> implements FeatureExtractor<LabeledData<Supplier<Image>, L>> {

	private int featureCount;

	public LabeledImageSupplierFeatureExtractor(int featureCount) {
		this.featureCount = featureCount;
	}

	@Override
	public float[] getFeatures(LabeledData<Supplier<Image>, L> data) throws FeatureExtractionException {
		try {
			Image image = data.getData().get();
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
