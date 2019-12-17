package org.ml4j.nn.datasets.images;

import java.util.function.Supplier;

import org.ml4j.images.Image;
import org.ml4j.nn.datasets.DataBatchImpl;

public class ImagesBatchImpl extends DataBatchImpl<Supplier<Image>> implements ImagesBatch {

	public ImagesBatchImpl(int batchSize) {
		super(batchSize);
	}
}
