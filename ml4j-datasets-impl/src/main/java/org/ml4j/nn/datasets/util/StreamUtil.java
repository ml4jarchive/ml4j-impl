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
package org.ml4j.nn.datasets.util;

import java.io.Closeable;
import java.io.EOFException;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.UncheckedIOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.function.Function;
import java.util.function.Supplier;
import java.util.stream.Stream;

import org.ml4j.images.Image;
import org.ml4j.nn.datasets.DataBatch;
import org.ml4j.nn.datasets.DataBatchImpl;
import org.ml4j.nn.datasets.LabeledData;
import org.ml4j.nn.datasets.LabeledDataBatchImpl;
import org.ml4j.nn.datasets.LabeledDataImpl;
import org.ml4j.nn.datasets.LabeledDataSet;
import org.ml4j.nn.datasets.images.ImagesBatch;
import org.ml4j.nn.datasets.images.ImagesBatchImpl;
import org.ml4j.nn.datasets.images.ImagesDataSet;
import org.ml4j.nn.datasets.images.LabeledImagesDataBatch;
import org.ml4j.nn.datasets.images.LabeledImagesDataBatchImpl;
import org.ml4j.nn.datasets.images.LabeledImagesDataSet;

public class StreamUtil {

	public static <T> Stream<DataBatch<T>> partition(Stream<T> stream, int batchSize) {
		List<DataBatch<T>> currentBatch = new ArrayList<>(); // just to make it mutable
		currentBatch.add(new DataBatchImpl<T>(batchSize));
		return Stream.concat(stream.sequential().map(new Function<T, DataBatch<T>>() {
			public DataBatch<T> apply(T t) {
				currentBatch.get(0).add(t);
				return currentBatch.get(0).size() == batchSize ? currentBatch.set(0, new DataBatchImpl<>(batchSize))
						: null;
			}
		}), Stream.generate(() -> currentBatch.get(0).isEmpty() ? null : currentBatch.get(0)).limit(1))
				.filter(Objects::nonNull);
	}

	/*
	 * public static <T, L> Stream<LabeledDataBatch<T, L>>
	 * partition(LabeledDataSet<T, L> labeledDataSet, int batchSize){
	 * List<LabeledDataBatch<T, L>> currentBatch = new ArrayList<>(); //just to make
	 * it mutable currentBatch.add(new LabeledDataBatchImpl<T, L>(batchSize));
	 * return Stream.concat(labeledDataSet.getLabeledData() .sequential() .map(new
	 * Function<LabeledData<T, L>, LabeledDataBatch<T, L>>(){ public
	 * LabeledDataBatch<T, L> apply(LabeledData<T, L> t){
	 * currentBatch.get(0).add(t); return currentBatch.get(0).size() == batchSize ?
	 * currentBatch.set(0,new LabeledDataBatchImpl<>(batchSize)): null; } }),
	 * Stream.generate(()->currentBatch.get(0).isEmpty()?null:currentBatch.get(0))
	 * .limit(1) ).filter(Objects::nonNull); }
	 */

	/*
	 * public static <T, L> Stream<FloatArrayLabeledDataBatch>
	 * partition2(LabeledDataSet<T, L> labeledDataSet, FeatureExtractor<T>
	 * featureExtractor, LabelMapper<L> labelMapper, int batchSize){
	 * List<FloatArrayLabeledDataBatch> currentBatch = new ArrayList<>(); //just to
	 * make it mutable currentBatch.add(new FloatArrayLabeledDataBatchImpl<T,
	 * L>(batchSize)); return Stream.concat(labeledDataSet.getLabeledData()
	 * .sequential() .map(new Function<LabeledData<T, L>,
	 * FloatArrayLabeledDataBatch>(){ public FloatArrayLabeledDataBatch
	 * apply(LabeledData<T, L> t){ currentBatch.get(0).add(new
	 * LabeledDataImpl<>(featureExtractor.getFeatures(t.getData()),
	 * labelMapper.getAsFloatArray(t.getLabel()))); return
	 * currentBatch.get(0).size() == batchSize ? currentBatch.set(0,new
	 * FloatArrayLabeledDataBatchImpl<>(batchSize)): null; } }),
	 * Stream.generate(()->currentBatch.get(0).isEmpty()?null:currentBatch.get(0))
	 * .limit(1) ).filter(Objects::nonNull); }
	 */

	public static <T, L> Stream<LabeledImagesDataBatch<L>> partition3(LabeledImagesDataSet<L> labeledDataSet,
			int batchSize) {
		List<LabeledImagesDataBatchImpl<L>> currentBatch = new ArrayList<>(); // just to make it mutable
		currentBatch.add(new LabeledImagesDataBatchImpl<L>(batchSize));
		return Stream.concat(labeledDataSet.stream().sequential()
				.map(new Function<LabeledData<Supplier<Image>, L>, LabeledImagesDataBatch<L>>() {
					public LabeledImagesDataBatch<L> apply(LabeledData<Supplier<Image>, L> t) {
						currentBatch.get(0).add(new LabeledDataImpl<Supplier<Image>, L>(t.getData(), t.getLabel()));
						return currentBatch.get(0).size() == batchSize
								? currentBatch.set(0, new LabeledImagesDataBatchImpl<>(batchSize))
								: null;
					}
				}), Stream.generate(() -> currentBatch.get(0).isEmpty() ? null : currentBatch.get(0)).limit(1))
				.filter(Objects::nonNull);
	}

	public static <T, L> Stream<ImagesBatch> partition4(ImagesDataSet imagesDataSet, int batchSize) {
		List<ImagesBatchImpl> currentBatch = new ArrayList<>(); // just to make it mutable
		currentBatch.add(new ImagesBatchImpl(batchSize));
		return Stream.concat(imagesDataSet.stream().sequential().map(new Function<Supplier<Image>, ImagesBatch>() {
			public ImagesBatch apply(Supplier<Image> t) {
				currentBatch.get(0).add(t);
				return currentBatch.get(0).size() == batchSize ? currentBatch.set(0, new ImagesBatchImpl(batchSize))
						: null;
			}
		}), Stream.generate(() -> currentBatch.get(0).isEmpty() ? null : currentBatch.get(0)).limit(1))
				.filter(Objects::nonNull);
	}

	public static <T, L> Stream<LabeledDataBatchImpl<T, L>> partition4(LabeledDataSet<T, L> labeledDataSet,
			int batchSize) {
		List<LabeledDataBatchImpl<T, L>> currentBatch = new ArrayList<>(); // just to make it mutable
		currentBatch.add(new LabeledDataBatchImpl<>(batchSize));
		return Stream.concat(
				labeledDataSet.stream().sequential().map(new Function<LabeledData<T, L>, LabeledDataBatchImpl<T, L>>() {
					public LabeledDataBatchImpl<T, L> apply(LabeledData<T, L> t) {
						currentBatch.get(0).add(t);
						return currentBatch.get(0).size() == batchSize
								? currentBatch.set(0, new LabeledDataBatchImpl<T, L>(batchSize))
								: null;
					}
				}), Stream.generate(() -> currentBatch.get(0).isEmpty() ? null : currentBatch.get(0)).limit(1))
				.filter(Objects::nonNull);
	}
	
	
	public static <T> Stream<T> toStream(final ObjectInputStream stream, final Class<T> cls) {
		return Stream.generate(() -> cls.cast(readObject(stream))).onClose(() -> close(stream)).takeWhile(e -> e != null);
	}
	
	private static Object readObject(ObjectInputStream stream) {
	    try {
	        Object o = stream.readObject();
	        if (o == null) {
	        	throw new IllegalStateException("Objects in the input stream cannot be null");
	        }
	        return o;
	        
	    } catch (EOFException e) {
	        return null;
	    }
	    catch (IOException e) {
	        throw new UncheckedIOException(e);
	    }
	    catch (ClassNotFoundException e) {
	        throw new RuntimeException(e);
	    }
	}

	private static void close(Closeable c) {
	    try {
	        c.close();
	    } catch (IOException e) {
	    	// TODO
	        // logger.log(Level.WARNING, "Couldn't close " + c, e);
	    }
	}

	
}
