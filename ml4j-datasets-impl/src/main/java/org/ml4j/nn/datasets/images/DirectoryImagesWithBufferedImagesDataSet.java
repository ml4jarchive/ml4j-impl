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
package org.ml4j.nn.datasets.images;

import java.awt.image.BufferedImage;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.List;
import java.util.function.Predicate;
import java.util.function.Supplier;
import java.util.stream.Stream;

import javax.imageio.ImageIO;

import org.ml4j.images.Image;
import org.ml4j.images.MultiChannelImage;
import org.ml4j.nn.datasets.LabeledData;
import org.ml4j.nn.datasets.LabeledDataImpl;
import org.ml4j.nn.datasets.LabeledDataSetImpl;
import org.ml4j.nn.datasets.exceptions.FeatureExtractionRuntimeException;
import org.ml4j.nn.datasets.featureextraction.BufferedImageFeatureExtractor;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class DirectoryImagesWithBufferedImagesDataSet extends LabeledDataSetImpl<Supplier<Image>, Supplier<BufferedImage>>
		implements ImagesWithBufferedImagesDataSet {

	private static final Logger LOGGER = LoggerFactory.getLogger(DirectoryImagesWithBufferedImagesDataSet.class);

	public DirectoryImagesWithBufferedImagesDataSet(Path directory, Predicate<Path> pathPredicate) {
		super(() -> getImages(directory, pathPredicate, null, null));
	}

	public DirectoryImagesWithBufferedImagesDataSet(Path directory, Predicate<Path> pathPredicate, int rescaleWidth,
			int rescaleHeight) {
		super(() -> getImages(directory, pathPredicate, rescaleWidth, rescaleHeight));
	}

	private static Stream<LabeledData<Supplier<Image>, Supplier<BufferedImage>>> getImages(Path file, Predicate<Path> pathPredicate,
			Integer rescaleWidth, Integer rescaleHeight) {
		try {
			if (Files.isDirectory(file)) {
				return Files.list(file).filter(f -> pathPredicate.test(f))
						.flatMap(f -> getImages(f, pathPredicate, rescaleWidth, rescaleHeight));
			} else {
				Supplier<LabeledData<Image, BufferedImage>> imageDetailsSupplier = getImage(file, rescaleWidth, rescaleHeight);
				List<LabeledData<Supplier<Image>, Supplier<BufferedImage>>> list = Arrays
						.asList(new ImageDetailsLabeledData(
								imageDetailsSupplier));
				return list.stream();
			}

		} catch (IOException e) {
			throw new RuntimeException(e);
		}
	}

	@Override
	public ImagesDataSet getDataSet() {
		return getDataSet();
	}
	
	private static class ImageDetailsLabeledData implements LabeledData<Supplier<Image>, Supplier<BufferedImage>> {

		private Supplier<LabeledData<Image, BufferedImage>> detailsSupplier;
		private LabeledData<Image, BufferedImage> imageDetails;
		
		public ImageDetailsLabeledData(Supplier<LabeledData<Image, BufferedImage>> detailsSupplier) {
			this.detailsSupplier = detailsSupplier;
		}
		
		@Override
		public Supplier<Image> getData() {
			if (imageDetails == null) {
				imageDetails = detailsSupplier.get();
			}
			return () -> imageDetails.getData();
		}

		@Override
		public Supplier<BufferedImage> getLabel() {
			if (imageDetails == null) {
				imageDetails = detailsSupplier.get();
			}
			return () -> imageDetails.getLabel();
		}
	}

	private static Supplier<LabeledData<Image, BufferedImage>> getImage(Path path, Integer rescaleWidth, Integer rescaleHeight) {
		return () -> getImageFromFile(path, rescaleWidth, rescaleHeight);
	}

	private static LabeledData<Image, BufferedImage> getImageFromFile(Path path, Integer rescaleWidth, Integer rescaleHeight) {

		try {
			LOGGER.debug("Loading image from file:" + path.toFile());
			BufferedImage bufferedImage = ImageIO.read(path.toFile());
			if (bufferedImage != null) {
				int width = rescaleWidth != null ? rescaleWidth : bufferedImage.getWidth();
				int height = rescaleHeight != null ? rescaleHeight : bufferedImage.getHeight();
				BufferedImageFeatureExtractor mapper = new BufferedImageFeatureExtractor(width, height);
				return new LabeledDataImpl<>(new MultiChannelImage(mapper.getFeatures(bufferedImage), 3, width, height, 0, 0), 
						bufferedImage);

			} else {
				throw new IllegalArgumentException("Unable to read file" + path);
			}
		} catch (Exception e) {
			throw new FeatureExtractionRuntimeException("Unable to read features from file:" + path, e);
		}
	}

}
