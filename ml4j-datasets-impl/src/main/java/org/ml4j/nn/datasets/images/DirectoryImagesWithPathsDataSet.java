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
import org.ml4j.nn.datasets.DataLabeler;
import org.ml4j.nn.datasets.DataSet;
import org.ml4j.nn.datasets.DataSetImpl;
import org.ml4j.nn.datasets.LabeledData;
import org.ml4j.nn.datasets.LabeledDataImpl;
import org.ml4j.nn.datasets.LabeledDataSetImpl;
import org.ml4j.nn.datasets.exceptions.FeatureExtractionRuntimeException;
import org.ml4j.nn.datasets.featureextraction.BufferedImageFeatureExtractor;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class DirectoryImagesWithPathsDataSet extends LabeledDataSetImpl<Supplier<Image>, Path>
		implements ImagesWithPathsDataSet {

	private static final Logger LOGGER = LoggerFactory.getLogger(DirectoryImagesWithPathsDataSet.class);

	
	public DirectoryImagesWithPathsDataSet(Path directory, Predicate<Path> pathPredicate) {
		super(() -> getImages(directory, pathPredicate, null, null));
	}
	
	public DirectoryImagesWithPathsDataSet(Path directory, Predicate<Path> pathPredicate, int rescaleWidth, int rescaleHeight) {
		super(() -> getImages(directory, pathPredicate, rescaleWidth, rescaleHeight));
	}

	private static Stream<LabeledData<Supplier<Image>, Path>> getImages(Path file, Predicate<Path> pathPredicate, Integer rescaleWidth, Integer rescaleHeight) {
		try {
			if (Files.isDirectory(file)) {
				return Files.list(file).filter(f -> pathPredicate.test(f)).flatMap(f -> getImages(f, pathPredicate, rescaleWidth, rescaleHeight));
			} else {
				List<LabeledData<Supplier<Image>, Path>> list = Arrays
						.asList(new LabeledDataImpl<>(getImage(file, rescaleWidth, rescaleHeight), file));
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

	private static Supplier<Image> getImage(Path path, Integer rescaleWidth, Integer rescaleHeight) {
		return () -> getImageFromFile(path, rescaleWidth, rescaleHeight);
	}

	private static Image getImageFromFile(Path path, Integer rescaleWidth, Integer rescaleHeight) {
		
		try {
			LOGGER.debug("Loading image from file:" + path.toFile());
			BufferedImage bufferedImage = ImageIO.read(path.toFile());
			if (bufferedImage != null) {
				int width = rescaleWidth != null ? rescaleWidth :
					bufferedImage.getWidth();
				int height = rescaleHeight != null ? rescaleHeight : bufferedImage.getHeight();
				BufferedImageFeatureExtractor mapper = new BufferedImageFeatureExtractor(width, height);
				return new MultiChannelImage(mapper.getFeatures(bufferedImage), 3, width, height, 0, 0);

			} else {
				throw new IllegalArgumentException("Unable to read file" + path);
			}
		} catch (Exception e) {
			throw new FeatureExtractionRuntimeException("Unable to read features from file:" + path, e);
		} 
	}
	
	@Override
	public DataSet<Path> getPathsDataSet() {
		return new DataSetImpl<>(() -> getLabels());
	}

	public <L> LabeledImagesDataSet<L> getLabeledImagesDataSet(DataLabeler<Path, L> pathBasedLabeler) {
		return new DirectoryImagesLabeledDataSet<L>(this, pathBasedLabeler);
	}

}
