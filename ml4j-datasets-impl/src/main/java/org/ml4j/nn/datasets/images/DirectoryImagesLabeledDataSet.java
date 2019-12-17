package org.ml4j.nn.datasets.images;

import java.nio.file.Path;
import java.util.function.Supplier;

import org.ml4j.images.Image;
import org.ml4j.nn.datasets.DataLabeler;
import org.ml4j.nn.datasets.DataSetImpl;
import org.ml4j.nn.datasets.LabeledDataSetImpl;

public class DirectoryImagesLabeledDataSet<L> extends LabeledDataSetImpl<Supplier<Image>, L>
		implements LabeledImagesDataSet<L> {

	public DirectoryImagesLabeledDataSet(DirectoryImagesWithPathsDataSet labeledDataSet,
			DataLabeler<Path, L> dataLabeler) {
		super(new DataSetImpl<>(() -> labeledDataSet.stream()), dataLabeler);
	}

	@Override
	public ImagesDataSet getDataSet() {
		return new ImagesDataSetImpl(() -> super.getDataSet().stream());
	}
}
