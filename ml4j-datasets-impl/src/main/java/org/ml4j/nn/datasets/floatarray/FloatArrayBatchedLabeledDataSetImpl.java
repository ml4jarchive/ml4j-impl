package org.ml4j.nn.datasets.floatarray;

import java.util.function.Supplier;
import java.util.stream.Stream;

import org.ml4j.MatrixFactory;
import org.ml4j.nn.datasets.BatchedLabeledDataSetImpl;
import org.ml4j.nn.datasets.DataBatch;
import org.ml4j.nn.datasets.DataBatchImpl;
import org.ml4j.nn.datasets.LabeledData;
import org.ml4j.nn.datasets.LabeledDataImpl;
import org.ml4j.nn.datasets.neuronsactivation.NeuronsActivationLabeledDataSet;
import org.ml4j.nn.datasets.neuronsactivation.NeuronsActivationLabeledDataSetImpl;
import org.ml4j.nn.neurons.NeuronsActivation;
import org.ml4j.streams.Streamable;

public class FloatArrayBatchedLabeledDataSetImpl extends BatchedLabeledDataSetImpl<float[], float[]>
		implements FloatArrayBatchedLabeledDataSet {

    private int featureCount;
    private int labelFeatureCount;
			
			
	public FloatArrayBatchedLabeledDataSetImpl(
			Supplier<Stream<DataBatch<LabeledData<float[], float[]>>>> dataSupplier, int featureCount, int labelFeatureCount) {
		super(dataSupplier);
		this.featureCount = featureCount;
		this.labelFeatureCount = labelFeatureCount;
		if (featureCount == 0) {
			throw new IllegalArgumentException();
		}
		if (labelFeatureCount == 0) {
			throw new IllegalArgumentException();
		}
	}

	@Override
	public NeuronsActivationLabeledDataSet toNeuronsActivationLabeledDataSet(MatrixFactory matrixFactory) {
		return new NeuronsActivationLabeledDataSetImpl(
				() -> stream().map(batch -> createLabeledData(matrixFactory, batch)));
	}


	private LabeledData<NeuronsActivation, NeuronsActivation> createLabeledData(MatrixFactory matrixFactory,
			DataBatch<LabeledData<float[], float[]>> batch) {
		return createFloatArrayLabeledDataBatch(batch).toNeuronsActivations(matrixFactory);
	}
			
	
	private FloatArrayLabeledDataBatch createFloatArrayLabeledDataBatch(DataBatch<LabeledData<float[], float[]>> batch) {
	
		LabeledData<DataBatch<float[]>, DataBatch<float[]>> dataBatches = convertBatchOfLabeledDataToLabeledDataOfBatches(batch);
		
		return new FloatArrayLabeledDataBatchImpl(dataBatches.getData(), dataBatches.getLabel(), featureCount, labelFeatureCount);
	}

	
	private <E, L> LabeledData<DataBatch<E>, DataBatch<L>> convertBatchOfLabeledDataToLabeledDataOfBatches(
			DataBatch<LabeledData<E, L>> dataBatchOfLabeledData) {

		Streamable<E> data = dataBatchOfLabeledData.map(batch -> batch.getData());
		Streamable<L> labels = dataBatchOfLabeledData.map(batch -> batch.getLabel());

		DataBatch<E> dataBatch = new DataBatchImpl<>(data.stream(), dataBatchOfLabeledData.size());
		DataBatch<L> labelsBatch = new DataBatchImpl<>(labels.stream(), dataBatchOfLabeledData.size());

		return new LabeledDataImpl<>(dataBatch, labelsBatch);
	}

}
