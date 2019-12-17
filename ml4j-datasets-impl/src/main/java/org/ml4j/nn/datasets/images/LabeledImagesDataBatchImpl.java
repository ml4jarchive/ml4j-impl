package org.ml4j.nn.datasets.images;

import java.util.function.Supplier;

import org.ml4j.images.Image;
import org.ml4j.nn.datasets.DataBatch;
import org.ml4j.nn.datasets.LabeledDataBatchImpl;

public class LabeledImagesDataBatchImpl<L> extends LabeledDataBatchImpl<Supplier<Image>, L>
		implements LabeledImagesDataBatch<L> {

	public LabeledImagesDataBatchImpl(DataBatch<Supplier<Image>> dataBatch, DataBatch<L> labelBatch) {
		super(dataBatch, labelBatch);
	}

	public LabeledImagesDataBatchImpl(int batchSize) {
		super(batchSize);
	}
}
