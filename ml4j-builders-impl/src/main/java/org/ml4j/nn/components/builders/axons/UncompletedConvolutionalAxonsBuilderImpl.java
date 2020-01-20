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
package org.ml4j.nn.components.builders.axons;

import java.util.function.Consumer;
import java.util.function.Supplier;

import org.ml4j.Matrix;
import org.ml4j.nn.axons.AxonsContext;
import org.ml4j.nn.neurons.Neurons3D;

public class UncompletedConvolutionalAxonsBuilderImpl<C extends Axons3DBuilder<?>>
		extends UncompletedAxonsBuilderImpl<Neurons3D, C> implements UncompletedConvolutionalAxonsBuilder<C> {

	private int strideWidth = 1;
	private int strideHeight = 1;
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
		// P = ((S-1)*W-S+F)/2
		if (filterWidth != null) {
			double paddingWidth = ((double) ((strideWidth - 1) * this.getLeftNeurons().getWidth() - strideWidth
					+ filterWidth)) / 2d;
			int paddingWidthInt = (int) paddingWidth;
			if (paddingWidth != paddingWidthInt) {
				paddingWidthInt = paddingWidthInt + 1;
			}
			this.paddingWidth = paddingWidthInt;

		}
		if (filterHeight != null) {
			double paddingHeight = ((double) ((strideHeight - 1) * this.getLeftNeurons().getHeight() - strideHeight
					+ filterHeight)) / 2d;
			int paddingHeightInt = (int) paddingHeight;
			if (paddingHeight != paddingHeightInt) {
				paddingHeightInt = paddingHeightInt + 1;
			}
			this.paddingHeight = paddingHeightInt;
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
	public UncompletedConvolutionalAxonsBuilder<C> withAxonsContextConfigurer(
			Consumer<AxonsContext> axonsContextConfigurer) {
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
