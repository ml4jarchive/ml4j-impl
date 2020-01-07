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
