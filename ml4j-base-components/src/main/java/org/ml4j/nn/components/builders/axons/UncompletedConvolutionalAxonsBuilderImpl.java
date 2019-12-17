package org.ml4j.nn.components.builders.axons;

import java.util.function.Consumer;
import java.util.function.Supplier;

import org.ml4j.Matrix;
import org.ml4j.nn.axons.AxonsContext;
import org.ml4j.nn.components.DirectedComponentsContext;
import org.ml4j.nn.neurons.Neurons3D;

public class UncompletedConvolutionalAxonsBuilderImpl<C extends Axons3DBuilder> extends UncompletedAxonsBuilderImpl<Neurons3D, C> implements UncompletedConvolutionalAxonsBuilder<C> {

	private int strideWidth = 1;
	private int strideHeight= 1;
	private int paddingWidth = 0;
	private int paddingHeight = 0;
	private Integer filterWidth;
	private Integer filterHeight;
	
	public UncompletedConvolutionalAxonsBuilderImpl(Supplier<C> previousBuilder, Neurons3D leftNeurons) {
		super(previousBuilder, leftNeurons);
	}

	@Override
	public C withConnectionToNeurons(Neurons3D neurons) {
		previousBuilderSupplier.get().getComponentsGraphNeurons().setRightNeurons(neurons);
		return previousBuilderSupplier.get();
	}
	
	@Override
	public UncompletedConvolutionalAxonsBuilder<C> withConnectionWeights(Matrix connectionWeights) {
		previousBuilderSupplier.get().getBuilderState().setConnectionWeights(connectionWeights);
		return this;
	}
	
	@Override
	public UncompletedConvolutionalAxonsBuilder<C> withBiases(Matrix biases) {
		previousBuilderSupplier.get().getBuilderState().setBiases(biases);
		previousBuilderSupplier.get().getBuilderState().getComponentsGraphNeurons().setHasBiasUnit(true);
		return this;
	}
	
	@Override
	public UncompletedConvolutionalAxonsBuilder<C> withBiasUnit() {
		previousBuilderSupplier.get().getBuilderState().getComponentsGraphNeurons().setHasBiasUnit(true);
		return this;
	}

	@Override
	public UncompletedConvolutionalAxonsBuilder<C> withFilterSize(int width, int height) {
		filterWidth = width;
		filterHeight = height;
		return this;
	}

	@Override
	public UncompletedConvolutionalAxonsBuilder<C> withFilterCount(int filterCount) {
		return this;
	}

	@Override
	public UncompletedConvolutionalAxonsBuilder<C> withSamePadding() {
		//P = ((S-1)*W-S+F)/2
		if (filterWidth != null) {
			double paddingWidth =  ((double)((strideWidth-1) * this.getLeftNeurons().getWidth() - strideWidth + filterWidth))/2d;
			int paddingWidthInt = (int)paddingWidth;
			if (paddingWidth != paddingWidthInt) {
				paddingWidthInt = paddingWidthInt + 1;
			}
			this.paddingWidth =paddingWidthInt;
			
		}
		if (filterHeight != null) {
			double paddingHeight =  ((double)((strideHeight-1) * this.getLeftNeurons().getHeight() - strideHeight + filterHeight))/2d;
			int paddingHeightInt = (int)paddingHeight;
			if (paddingHeight != paddingHeightInt) {
				paddingHeightInt = paddingHeightInt + 1;
			}
			this.paddingHeight =paddingHeightInt;
		}
		
		
		return this;
	}

	@Override
	public UncompletedConvolutionalAxonsBuilder<C> withStride(int widthStride, int heightStride) {
		this.strideWidth = widthStride;
		this.strideHeight = heightStride;
		return this;
	}

	@Override
	public UncompletedConvolutionalAxonsBuilder<C> withValidPadding() {
		return this;
	}

	@Override
	public UncompletedConvolutionalAxonsBuilder<C> withPadding(int widthPadding, int heightPadding) {
		this.paddingWidth = widthPadding;
		this.paddingHeight = heightPadding;
		return this;
	}
	
	@Override
	public UncompletedConvolutionalAxonsBuilder<C> withAxonsContext(
			DirectedComponentsContext directedComponentsContext, Consumer<AxonsContext> axonsContextConfigurer) {
		this.directedComponentsContext = directedComponentsContext;
		this.axonsContextConfigurer = axonsContextConfigurer;
		return this;
	}

	@Override
	public int getStrideWidth() {
		return strideWidth;
	}

	@Override
	public int getPaddingWidth() {
		return paddingWidth;
	}
	
	@Override
	public int getStrideHeight() {
		return strideHeight;
	}

	@Override
	public int getPaddingHeight() {
		return paddingHeight;
	}
}
