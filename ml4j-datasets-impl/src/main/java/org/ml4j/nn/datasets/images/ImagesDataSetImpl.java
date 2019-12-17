package org.ml4j.nn.datasets.images;

import java.util.function.Supplier;
import java.util.stream.Stream;

import org.ml4j.images.Image;
import org.ml4j.nn.datasets.DataSetImpl;

public class ImagesDataSetImpl extends DataSetImpl<Supplier<Image>> implements ImagesDataSet {

	public ImagesDataSetImpl(Supplier<Stream<Supplier<Image>>> dataSupplier) {
		super(dataSupplier);
	}

}
