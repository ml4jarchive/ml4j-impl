package org.ml4j.nn.datasets;

import java.util.function.Supplier;
import java.util.stream.Stream;

import org.ml4j.nn.datasets.util.StreamUtil;

public class DataSetImpl<E> implements DataSet<E> {

	private Supplier<Stream<E>> dataSupplier;

	public DataSetImpl(Supplier<Stream<E>> dataSupplier) {
		this.dataSupplier = dataSupplier;
	}

	@Override
	public Stream<E> stream() {
		return dataSupplier.get();
	}

	@Override
	public BatchedDataSet<E> toBatchedDataSet(int batchSize) {
		return new BatchedDataSetImpl<E>(() -> StreamUtil.partition(stream(), batchSize));
	}

}
