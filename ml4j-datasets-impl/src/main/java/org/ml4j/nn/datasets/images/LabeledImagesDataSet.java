package org.ml4j.nn.datasets.images;

import java.util.function.Supplier;

import org.ml4j.images.Image;
import org.ml4j.nn.datasets.LabeledDataSet;

public interface LabeledImagesDataSet<L> extends LabeledDataSet<Supplier<Image>, L> {

	ImagesDataSet getDataSet();
}
