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
package org.ml4j.nn.components.builders.base;

import java.util.ArrayList;
import java.util.List;

import org.ml4j.nn.activationfunctions.ActivationFunctionProperties;
import org.ml4j.nn.activationfunctions.ActivationFunctionType;
import org.ml4j.nn.activationfunctions.DifferentiableActivationFunction;
import org.ml4j.nn.axons.Axons3DConfig;
import org.ml4j.nn.axons.AxonsContext;
import org.ml4j.nn.axons.BatchNormConfig;
import org.ml4j.nn.axons.BiasMatrix;
import org.ml4j.nn.axons.WeightsMatrix;
import org.ml4j.nn.components.AxonsContextAwareNeuralComponent;
import org.ml4j.nn.components.DirectedComponentsContext;
import org.ml4j.nn.components.NeuralComponent;
import org.ml4j.nn.components.builders.Base3DGraphBuilderState;
import org.ml4j.nn.components.builders.axons.Axons3DBuilder;
import org.ml4j.nn.components.builders.axons.Axons3DPermitted;
import org.ml4j.nn.components.builders.axons.AxonsBuilder;
import org.ml4j.nn.components.builders.axons.UncompletedBatchNormAxons3DBuilder;
import org.ml4j.nn.components.builders.axons.UncompletedBatchNormAxons3DBuilderImpl;
import org.ml4j.nn.components.builders.axons.UncompletedConvolutionalAxonsBuilder;
import org.ml4j.nn.components.builders.axons.UncompletedConvolutionalAxonsBuilderImpl;
import org.ml4j.nn.components.builders.axons.UncompletedFullyConnectedAxonsBuilder;
import org.ml4j.nn.components.builders.axons.UncompletedFullyConnectedAxonsBuilderImpl;
import org.ml4j.nn.components.builders.axons.UncompletedPoolingAxonsBuilder;
import org.ml4j.nn.components.builders.axons.UncompletedPoolingAxonsBuilderImpl;
import org.ml4j.nn.components.builders.common.ComponentsContainer;
import org.ml4j.nn.components.builders.componentsgraph.ComponentsGraphNeurons;
import org.ml4j.nn.components.builders.synapses.Synapses3DPermitted;
import org.ml4j.nn.components.builders.synapses.SynapsesAxons3DGraphBuilder;
import org.ml4j.nn.components.builders.synapses.SynapsesAxons3DGraphBuilderImpl;
import org.ml4j.nn.components.factories.NeuralComponentFactory;
import org.ml4j.nn.neurons.Neurons;
import org.ml4j.nn.neurons.Neurons3D;

public abstract class Base3DGraphBuilderImpl<C extends Axons3DBuilder<T>, D extends AxonsBuilder<T>, T extends NeuralComponent<?>>
		implements Axons3DPermitted<C, D, T>, Synapses3DPermitted<C, D, T>, Axons3DBuilder<T> {

	protected NeuralComponentFactory<T> directedComponentFactory;

	protected Base3DGraphBuilderState initialBuilderState;

	private List<T> components;

	private List<T> chains;

	private List<Neurons3D> endNeurons;

	protected Base3DGraphBuilderState builderState;

	protected DirectedComponentsContext directedComponentsContext;

	public abstract C get3DBuilder();

	public abstract D getBuilder();

	public Base3DGraphBuilderState getBuilderState() {
		return builderState;
	}

	public WeightsMatrix getConnectionWeights() {
		return builderState.getConnectionWeights();
	}

	public Axons3DBuilder<T> withConnectionWeights(WeightsMatrix connectionWeights) {
		builderState.setConnectionWeights(connectionWeights);
		return this;
	}

	public BiasMatrix getBiases() {
		return builderState.getBiases();
	}

	public Axons3DBuilder<T> withBiases(BiasMatrix biases) {
		builderState.setBiases(biases);
		return this;
	}

	@Override
	public ComponentsContainer<Neurons, T> getAxonsBuilder() {
		return getBuilder();
	}

	@Override
	public List<T> getComponents() {
		addAxonsIfApplicable();
		return components;
	}

	@Override
	public ComponentsGraphNeurons<Neurons3D> getComponentsGraphNeurons() {
		return builderState.getComponentsGraphNeurons();
	}

	public Base3DGraphBuilderImpl(NeuralComponentFactory<T> directedComponentFactory,
			Base3DGraphBuilderState builderState, DirectedComponentsContext directedComponentsContext,
			List<T> components) {
		this.components = components;
		this.directedComponentFactory = directedComponentFactory;
		this.builderState = builderState;
		this.initialBuilderState = new Base3DGraphBuilderStateImpl(
				builderState.getComponentsGraphNeurons().getCurrentNeurons());
		this.endNeurons = new ArrayList<>();
		this.directedComponentsContext = directedComponentsContext;
		this.chains = new ArrayList<>();

	}

	public List<T> getChains() {
		return chains;
	}

	public List<Neurons3D> getEndNeurons() {
		return endNeurons;
	}

	public void addAxonsIfApplicable() {

		if ((builderState.getConvolutionalAxonsBuilder() != null)
				&& builderState.getComponentsGraphNeurons().getRightNeurons() != null) {
			Neurons3D leftNeurons = builderState.getComponentsGraphNeurons().getCurrentNeurons();
			if (builderState.getComponentsGraphNeurons().hasBiasUnit() && !leftNeurons.hasBiasUnit()) {
				leftNeurons = new Neurons3D(builderState.getComponentsGraphNeurons().getCurrentNeurons().getWidth(),
						builderState.getComponentsGraphNeurons().getCurrentNeurons().getHeight(),
						builderState.getComponentsGraphNeurons().getCurrentNeurons().getDepth(), true);
			}

			Axons3DConfig axons3DConfig = new Axons3DConfig(leftNeurons, builderState.getComponentsGraphNeurons().getRightNeurons())
					.withStrideWidth(builderState.getConvolutionalAxonsBuilder().getStrideWidth())
					.withStrideHeight(builderState.getConvolutionalAxonsBuilder().getStrideHeight())
					.withPaddingWidth(builderState.getConvolutionalAxonsBuilder().getPaddingWidth())
					.withPaddingHeight(builderState.getConvolutionalAxonsBuilder().getPaddingHeight());
			
			if (builderState.getConvolutionalAxonsBuilder().getFilterHeight() != null) {
				axons3DConfig = axons3DConfig.withFilterHeight(builderState.getConvolutionalAxonsBuilder().getFilterHeight());
			}
			
			if (builderState.getConvolutionalAxonsBuilder().getFilterWidth() != null) {
				axons3DConfig = axons3DConfig.withFilterWidth(builderState.getConvolutionalAxonsBuilder().getFilterWidth());
			}

			T axonsComponent = directedComponentFactory.createConvolutionalAxonsComponent(builderState.getConvolutionalAxonsBuilder().getName(), axons3DConfig,
					builderState.getConnectionWeights(), builderState.getBiases());

			if (builderState.getConvolutionalAxonsBuilder().getAxonsContextConfigurer() != null) {
				// TODO
				if (axonsComponent instanceof AxonsContextAwareNeuralComponent) {
					AxonsContext axonsContext = ((AxonsContextAwareNeuralComponent<?>) axonsComponent)
							.getContext(directedComponentsContext);
					builderState.getConvolutionalAxonsBuilder().getAxonsContextConfigurer().accept(axonsContext);
				}
			}

			components.add(axonsComponent);

			builderState.setConvolutionalAxonsBuilder(null);
			builderState.getComponentsGraphNeurons()
					.setCurrentNeurons(builderState.getComponentsGraphNeurons().getRightNeurons());
			builderState.getComponentsGraphNeurons().setRightNeurons(null);
			builderState.setConnectionWeights(null);
		}
		if ((builderState.getMaxPoolingAxonsBuilder() != null)
				&& builderState.getComponentsGraphNeurons().getRightNeurons() != null) {
			Axons3DConfig axons3DConfig = new Axons3DConfig(builderState.getComponentsGraphNeurons().getCurrentNeurons(), builderState.getComponentsGraphNeurons().getRightNeurons())
					.withStrideWidth(builderState.getMaxPoolingAxonsBuilder().getStrideWidth())
					.withStrideHeight(builderState.getMaxPoolingAxonsBuilder().getStrideHeight())
					.withPaddingWidth(builderState.getMaxPoolingAxonsBuilder().getPaddingWidth())
					.withPaddingHeight(builderState.getMaxPoolingAxonsBuilder().getPaddingHeight());
			
			if (builderState.getMaxPoolingAxonsBuilder().getFilterHeight() != null) {
				axons3DConfig = axons3DConfig.withFilterHeight(builderState.getMaxPoolingAxonsBuilder().getFilterHeight());
			}
			
			if (builderState.getMaxPoolingAxonsBuilder().getFilterWidth() != null) {
				axons3DConfig = axons3DConfig.withFilterWidth(builderState.getMaxPoolingAxonsBuilder().getFilterWidth());
			}
			
			T axonsComponent = directedComponentFactory.createMaxPoolingAxonsComponent(builderState.getMaxPoolingAxonsBuilder().getName(),
					axons3DConfig,
					builderState.getMaxPoolingAxonsBuilder().isScaleOutputs());
			this.components.add(axonsComponent);
			builderState.setMaxPoolingAxonsBuilder(null);
			builderState.getComponentsGraphNeurons()
					.setCurrentNeurons(builderState.getComponentsGraphNeurons().getRightNeurons());
			builderState.getComponentsGraphNeurons().setRightNeurons(null);
			builderState.setConnectionWeights(null);
		}
		if ((builderState.getBatchNormAxonsBuilder() != null)
				&& builderState.getComponentsGraphNeurons().getRightNeurons() != null) {
			Neurons3D leftNeurons = builderState.getComponentsGraphNeurons().getCurrentNeurons();
			if (builderState.getComponentsGraphNeurons().hasBiasUnit() && !leftNeurons.hasBiasUnit()) {
				leftNeurons = new Neurons3D(builderState.getComponentsGraphNeurons().getCurrentNeurons().getWidth(),
						builderState.getComponentsGraphNeurons().getCurrentNeurons().getHeight(),
						builderState.getComponentsGraphNeurons().getCurrentNeurons().getDepth(), true);
			}

			T axonsComponent = directedComponentFactory.createBatchNormAxonsComponent(builderState.getBatchNormAxonsBuilder().getName(),
					builderState.getComponentsGraphNeurons().getRightNeurons(),
					new BatchNormConfig<>(builderState.getBatchNormAxonsBuilder().getBatchNormDimension())
					.withGammaColumnVector(builderState.getBatchNormAxonsBuilder().getGamma())
					.withBetaColumnVector(builderState.getBatchNormAxonsBuilder().getBeta())
					.withMeanColumnVector(builderState.getBatchNormAxonsBuilder().getMean())
					.withVarianceColumnVector(builderState.getBatchNormAxonsBuilder().getVariance()));

			if (builderState.getBatchNormAxonsBuilder().getAxonsContextConfigurer() != null) {
				// TODO
				if (axonsComponent instanceof AxonsContextAwareNeuralComponent) {
					AxonsContext axonsContext = ((AxonsContextAwareNeuralComponent<?>) axonsComponent)
							.getContext(directedComponentsContext);
					builderState.getBatchNormAxonsBuilder().getAxonsContextConfigurer().accept(axonsContext);
				}
			}

			this.components.add(axonsComponent);
			builderState.setBatchNormAxonsBuilder(null);
			builderState.getComponentsGraphNeurons()
					.setCurrentNeurons(builderState.getComponentsGraphNeurons().getRightNeurons());
			builderState.getComponentsGraphNeurons().setRightNeurons(null);
			builderState.setConnectionWeights(null);
		}
		if ((builderState.getAveragePoolingAxonsBuilder() != null)
				&& builderState.getComponentsGraphNeurons().getRightNeurons() != null) {
			Axons3DConfig axons3DConfig = new Axons3DConfig(builderState.getComponentsGraphNeurons().getCurrentNeurons(),
					builderState.getComponentsGraphNeurons().getRightNeurons())
					.withStrideWidth(builderState.getAveragePoolingAxonsBuilder().getStrideWidth())
					.withStrideHeight(builderState.getAveragePoolingAxonsBuilder().getStrideHeight())
					.withPaddingWidth(builderState.getAveragePoolingAxonsBuilder().getPaddingWidth())
					.withPaddingHeight(builderState.getAveragePoolingAxonsBuilder().getPaddingHeight());
			
			if (builderState.getAveragePoolingAxonsBuilder().getFilterHeight() != null) {
				axons3DConfig = axons3DConfig.withFilterHeight(builderState.getAveragePoolingAxonsBuilder().getFilterHeight());
			}
			
			if (builderState.getAveragePoolingAxonsBuilder().getFilterWidth() != null) {
				axons3DConfig = axons3DConfig.withFilterWidth(builderState.getAveragePoolingAxonsBuilder().getFilterWidth());
			}
			
			T axonsComponent = directedComponentFactory.createAveragePoolingAxonsComponent(builderState.getAveragePoolingAxonsBuilder().getName(),
					axons3DConfig);
			this.components.add(axonsComponent);
			builderState.setAveragePoolingAxonsBuilder(null);
			builderState.getComponentsGraphNeurons()
					.setCurrentNeurons(builderState.getComponentsGraphNeurons().getRightNeurons());
			builderState.getComponentsGraphNeurons().setRightNeurons(null);
			builderState.setConnectionWeights(null);
		}
	}

	public Axons3DBuilder<T> withBiasUnit() {
		builderState.getComponentsGraphNeurons().setHasBiasUnit(true);
		return this;
	}

	@Override
	public UncompletedFullyConnectedAxonsBuilder<D> withFullyConnectedAxons(String name) {
		addAxonsIfApplicable();
		builderState.setConnectionWeights(null);
		builderState.getComponentsGraphNeurons().setHasBiasUnit(false);
		UncompletedFullyConnectedAxonsBuilder<D> axonsBuilder = new UncompletedFullyConnectedAxonsBuilderImpl<>(name, 
				this::getBuilder, builderState.getComponentsGraphNeurons().getCurrentNeurons());
		builderState.setFullyConnectedAxonsBuilder(axonsBuilder);
		return axonsBuilder;
	}

	@Override
	public UncompletedPoolingAxonsBuilder<C> withMaxPoolingAxons(String name) {
		addAxonsIfApplicable();
		UncompletedPoolingAxonsBuilder<C> axonsBuilder = new UncompletedPoolingAxonsBuilderImpl<>(name, this::get3DBuilder,
				builderState.getComponentsGraphNeurons().getCurrentNeurons());
		builderState.setMaxPoolingAxonsBuilder(axonsBuilder);
		builderState.setConnectionWeights(null);
		builderState.getComponentsGraphNeurons().setHasBiasUnit(false);
		return axonsBuilder;
	}

	@Override
	public UncompletedPoolingAxonsBuilder<C> withAveragePoolingAxons(String name) {
		addAxonsIfApplicable();
		UncompletedPoolingAxonsBuilder<C> axonsBuilder = new UncompletedPoolingAxonsBuilderImpl<>(name, this::get3DBuilder,
				builderState.getComponentsGraphNeurons().getCurrentNeurons());
		builderState.setAveragePoolingAxonsBuilder(axonsBuilder);
		builderState.setConnectionWeights(null);
		builderState.getComponentsGraphNeurons().setHasBiasUnit(false);
		return axonsBuilder;
	}

	@Override
	public UncompletedBatchNormAxons3DBuilder<C> withBatchNormAxons(String name) {
		addAxonsIfApplicable();
		UncompletedBatchNormAxons3DBuilder<C> axonsBuilder = new UncompletedBatchNormAxons3DBuilderImpl<>(name,
				this::get3DBuilder, builderState.getComponentsGraphNeurons().getCurrentNeurons());
		builderState.setBatchNormAxonsBuilder(axonsBuilder);
		builderState.setConnectionWeights(null);
		builderState.getComponentsGraphNeurons().setHasBiasUnit(false);
		return axonsBuilder;
	}

	@Override
	public UncompletedConvolutionalAxonsBuilder<C> withConvolutionalAxons(String name) {
		addAxonsIfApplicable();
		UncompletedConvolutionalAxonsBuilder<C> axonsBuilder = new UncompletedConvolutionalAxonsBuilderImpl<>(name, 
				this::get3DBuilder, builderState.getComponentsGraphNeurons().getCurrentNeurons());
		builderState.setConvolutionalAxonsBuilder(axonsBuilder);
		builderState.setConnectionWeights(null);
		builderState.getComponentsGraphNeurons().setHasBiasUnit(false);
		return axonsBuilder;
	}

	@Override
	public SynapsesAxons3DGraphBuilder<C, D, T> withSynapses() {
		addAxonsIfApplicable();
		SynapsesAxons3DGraphBuilder<C, D, T> synapsesBuilder = new SynapsesAxons3DGraphBuilderImpl<>(this::get3DBuilder,
				this::getBuilder, directedComponentFactory, builderState, directedComponentsContext, new ArrayList<>());
		builderState.setSynapsesBuilder(synapsesBuilder);
		return synapsesBuilder;
	}

	public void addActivationFunction(String name, DifferentiableActivationFunction activationFunction) {
		addAxonsIfApplicable();
		components.add(directedComponentFactory.createDifferentiableActivationFunctionComponent(name,
				this.builderState.getComponentsGraphNeurons().getCurrentNeurons(), activationFunction));
	}

	public void addActivationFunction(String name, ActivationFunctionType activationFunctionType, ActivationFunctionProperties activationFunctionProperties) {
		addAxonsIfApplicable();
		components.add(directedComponentFactory.createDifferentiableActivationFunctionComponent(name,
				this.builderState.getComponentsGraphNeurons().getCurrentNeurons(), activationFunctionType, activationFunctionProperties));
	}

	public T getComponentChain() {
		addAxonsIfApplicable();
		return directedComponentFactory.createDirectedComponentChain(components);
	}

	@Override
	public void addComponents(List<T> components) {
		this.components.addAll(components);
	}

	@Override
	public void addComponent(T component) {
		this.components.add(component);
	}
}
