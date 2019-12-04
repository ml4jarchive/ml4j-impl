package org.ml4j.nn.axons;

import org.ml4j.MatrixFactory;
import org.ml4j.nn.neurons.Neurons3D;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public abstract class BasePoolingAxonsImpl<A extends PoolingAxons<A>> extends Axons3DBase<A>
		implements PoolingAxons<A> {

	/**
	 * Default serialization id.
	 */
	private static final long serialVersionUID = 1L;
	
	private static final Logger LOGGER = LoggerFactory.getLogger(BasePoolingAxonsImpl.class);

	public BasePoolingAxonsImpl(MatrixFactory matrixFactory, Neurons3D leftNeurons, Neurons3D rightNeurons,
			Axons3DConfig axonsConfig, boolean independentInputOutputChannels) {
		super(matrixFactory, leftNeurons, rightNeurons, axonsConfig, independentInputOutputChannels);
	}
	
	protected ConvolutionalFormatter createConvolutionalFormatter(int examples) {
		
		return new PoolingFormatterImpl3(leftNeurons, rightNeurons, getStrideWidth(), getStrideHeight(), getPaddingWidth(), getPaddingHeight(), examples, true);	
		// return new DummyMaxPoolingReformatterImpl(matrixFactory, examples, indexes.getColumns(), leftNeurons, rightNeurons, connectionWeightsMask, indexes);

	}
	
	@Override
	protected boolean isLeftInputDropoutSupported() {
		return false;
	}
	
	@Override
	public boolean isTrainable(AxonsContext axonsContext) {
		return false;
	}
}
