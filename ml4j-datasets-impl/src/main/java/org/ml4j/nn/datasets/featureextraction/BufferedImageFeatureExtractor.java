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

import java.awt.image.BufferedImage;
import java.io.IOException;

import org.ml4j.nn.datasets.FeatureExtractor;
import org.ml4j.nn.datasets.exceptions.FeatureExtractionException;

import net.coobird.thumbnailator.Thumbnails;

/**
 * Maps BufferedImage instances to float[]
 * 
 * @author Michael Lavelle
 *
 */
public class BufferedImageFeatureExtractor implements FeatureExtractor<BufferedImage> {

	private int width;
	private int height;

	public BufferedImageFeatureExtractor(int width, int height) {
		this.width = width;
		this.height = height;
	}

	public int getFeatureCount() {
		return width * height * 3;
	}

	public static BufferedImage resize(BufferedImage img, int newW, int newH) throws FeatureExtractionException {
		try {
			return Thumbnails.of(img).forceSize(newW, newH).asBufferedImage();
		} catch (IOException e) {
			throw new FeatureExtractionException("Unable to resize image", e);
		}
	}

	@Override
	public float[] getFeatures(BufferedImage image) throws FeatureExtractionException {

		if (image.getWidth() != width || image.getHeight() != height) {
			image = resize(image, width, height);
		}

		float[] data = new float[width * height * 3];

		int ind = 0;
		for (int w = 0; w < image.getWidth(); w++) {
			for (int h = 0; h < image.getHeight(); h++) {
				int color = image.getRGB(h, w);

				// extract each color component
				int red = (color >>> 16) & 0xFF;
				double redVal = ((double) red) / 255d;
				int green = (color >>> 8) & 0xFF;
				double greenVal = ((double) green) / 255d;
				int blue = (color >>> 0) & 0xFF;
				double blueVal = ((double) blue) / 255d;
				data[ind] = (float) redVal;
				data[ind + width * height] = (float) greenVal;
				data[ind + 2 * width * height] = (float) blueVal;
				ind++;
			}
		}

		return data;
	}

}
