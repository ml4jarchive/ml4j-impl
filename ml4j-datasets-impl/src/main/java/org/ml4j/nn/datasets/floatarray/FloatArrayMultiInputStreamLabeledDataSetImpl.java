package org.ml4j.nn.datasets.floatarray;

import java.io.File;

public class FloatArrayMultiInputStreamLabeledDataSetImpl extends FloatArrayMultiSourceLabeledDataSetImpl {

	public FloatArrayMultiInputStreamLabeledDataSetImpl(File dataFile, File labelFile, int featureCount, int labelFeatureCount) {
		super(new FloatArrayInputStreamDataSet(dataFile, featureCount), new FloatArrayInputStreamDataSet(labelFile, labelFeatureCount), featureCount, labelFeatureCount);
	}
	
	public FloatArrayMultiInputStreamLabeledDataSetImpl(String dataFileName, String labelFileName, int featureCount, int labelFeatureCount) {
		super(new FloatArrayInputStreamDataSet(new File(dataFileName), featureCount), new FloatArrayInputStreamDataSet(new File(labelFileName), labelFeatureCount), featureCount, labelFeatureCount);
	}

}
