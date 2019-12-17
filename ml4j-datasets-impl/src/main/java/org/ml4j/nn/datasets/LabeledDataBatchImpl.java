package org.ml4j.nn.datasets;

import java.util.Optional;
import java.util.stream.Stream;

import org.ml4j.nn.datasets.exceptions.FeatureExtractionException;
import org.ml4j.nn.datasets.exceptions.FeatureExtractionRuntimeException;
import org.ml4j.nn.datasets.floatarray.FloatArrayDataBatch;
import org.ml4j.nn.datasets.floatarray.FloatArrayDataBatchImpl;
import org.ml4j.nn.datasets.floatarray.FloatArrayLabeledDataBatch;
import org.ml4j.nn.datasets.floatarray.FloatArrayLabeledDataBatchImpl;
import org.ml4j.nn.datasets.floatarray.FloatArrayLabeledDataSet;
import org.ml4j.nn.datasets.floatarray.FloatArrayLabeledDataSetImpl;
import org.ml4j.nn.datasets.util.StreamUtil;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.codepoetics.protonpack.StreamUtils;


public class LabeledDataBatchImpl<E, L> implements LabeledDataBatch<E, L> {

	private static final Logger LOGGER = LoggerFactory.getLogger(LabeledDataBatchImpl.class);

	
	protected DataBatch<E> dataBatch;
	protected DataBatch<L> labelBatch;

	public LabeledDataBatchImpl(int batchSize) {
		this.dataBatch = new DataBatchImpl<>(batchSize);
		this.labelBatch = new DataBatchImpl<>(batchSize);
	}

	public LabeledDataBatchImpl(DataBatch<E> dataBatch, DataBatch<L> labelBatch) {
		this.dataBatch = dataBatch;
		this.labelBatch = labelBatch;
		if (dataBatch.size() != labelBatch.size()) {
			throw new IllegalArgumentException();
		}
	}

	@Override
	public DataBatch<E> getDataSet() {
		return dataBatch;
	}

	@Override
	public void add(E data, L label) {
		dataBatch.add(data);
		labelBatch.add(label);
	}

	@Override
	public Stream<L> getLabels() {
		return labelBatch.stream();
	}

	@Override
	public Stream<LabeledData<E, L>> stream() {
		return StreamUtils.zip(dataBatch.stream(), labelBatch.stream(), (l, r) -> createLabeledData(l, r)).filter(Optional::isPresent).map(optional -> optional.get());
	}

	public void add(LabeledData<E, L> labeledData) {
		dataBatch.add(labeledData.getData());
		labelBatch.add(labeledData.getLabel());
	}

	private Optional<LabeledData<E, L>> createLabeledData(E element, L label) {
		if (element == null || label == null) {
			return Optional.empty();
		} else {
			return Optional.of(new LabeledDataImpl<>(element, label));
		}
	}
	
	
	@Override
	public int size() {
		return dataBatch.size();
	}

	@Override
	public boolean isEmpty() {
		return dataBatch.isEmpty();
	}

	@Override
	public DataSet<L> getLabelsSet() {
		return labelBatch;
	}

	@Override
	public BatchedLabeledDataSet<E, L> toBatchedLabeledDataSet(int batchSize) {

		Stream<DataBatch<LabeledData<E, L>>> dataBatchStream = StreamUtil.partition(stream(), batchSize);

		return new BatchedLabeledDataSetImpl<E, L>(() -> dataBatchStream);
	}
	
	@Override
	public FloatArrayLabeledDataBatch toFloatArrayLabeledDataBatch(FeatureExtractor<E> featureExtractor, FeatureExtractor<L> labelMapper, FeatureExtractionErrorMode featureExtractionErrorMode) {		
		
		
		Stream<LabeledData<float[], float[]>> stream = stream().map(l -> getFeatures(l, featureExtractor, labelMapper, featureExtractionErrorMode)).filter(Optional::isPresent).map(Optional::get);
		FloatArrayDataBatch dataBatch = new FloatArrayDataBatchImpl(stream.map(l -> l.getData()), featureExtractor.getFeatureCount(), size());
		FloatArrayDataBatch labelBatch = new FloatArrayDataBatchImpl(stream.map(l -> l.getLabel()), featureExtractor.getFeatureCount(), size());
		
		return new FloatArrayLabeledDataBatchImpl(dataBatch, labelBatch, featureExtractor.getFeatureCount(), labelMapper.getFeatureCount());
	}

	@Override
	public FloatArrayLabeledDataSet toFloatArrayLabeledDataSet(FeatureExtractor<E> featureExtractor, FeatureExtractor<L> labelMapper,
			FeatureExtractionErrorMode featureExtractionErrorMode) {
		return new FloatArrayLabeledDataSetImpl(() -> 
			stream().map(l -> getFeatures(l, featureExtractor, labelMapper, featureExtractionErrorMode)).filter(Optional::isPresent).map(Optional::get),featureExtractor.getFeatureCount(), labelMapper.getFeatureCount());
	}

	
	private <T, S> Optional<LabeledData<float[], float[]>> getFeatures(LabeledData<T, S> element, FeatureExtractor<T> featureExtractor, FeatureExtractor<S> labelMapper, FeatureExtractionErrorMode featureExtractionErrorMode) {
		try {
			return Optional.of(new LabeledDataImpl<>(featureExtractor.getFeatures(element.getData()), labelMapper.getFeatures(element.getLabel())));
		} catch (FeatureExtractionException e) {
			if (featureExtractionErrorMode == FeatureExtractionErrorMode.LOG_WARNING) {
				LOGGER.warn("Ignoring data element due to feature extraction failure", e);
			} else if (featureExtractionErrorMode == FeatureExtractionErrorMode.RAISE_EXCEPTION) {
				throw new FeatureExtractionRuntimeException("Unable to obtain features", e);
			}
			return Optional.empty();
		}
	}
}
