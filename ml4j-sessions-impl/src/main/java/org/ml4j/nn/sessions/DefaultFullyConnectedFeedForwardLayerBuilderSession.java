package org.ml4j.nn.sessions;

import java.util.Arrays;
import java.util.function.Consumer;
import java.util.function.Supplier;

import org.ml4j.nn.activationfunctions.ActivationFunctionBaseType;
import org.ml4j.nn.activationfunctions.ActivationFunctionProperties;
import org.ml4j.nn.activationfunctions.ActivationFunctionType;
import org.ml4j.nn.activationfunctions.DifferentiableActivationFunction;
import org.ml4j.nn.axons.BatchNormConfig;
import org.ml4j.nn.axons.BatchNormConfig.BatchNormDimension;
import org.ml4j.nn.axons.BiasMatrix;
import org.ml4j.nn.axons.WeightsFormat;
import org.ml4j.nn.axons.WeightsFormatImpl;
import org.ml4j.nn.axons.WeightsMatrix;
import org.ml4j.nn.axons.WeightsMatrixImpl;
import org.ml4j.nn.axons.WeightsMatrixOrientation;
import org.ml4j.nn.layers.DirectedLayerFactory;
import org.ml4j.nn.layers.FullyConnectedFeedForwardLayer;
import org.ml4j.nn.layers.builders.FullyConnectedFeedForwardLayerPropertiesBuilder;
import org.ml4j.nn.layers.builders.FullyConnectedLayerAxonsConfig;
import org.ml4j.nn.layers.builders.FullyConnectedLayerConfigBuilder;
import org.ml4j.nn.neurons.Neurons;
import org.ml4j.nn.neurons.format.features.Dimension;

public class DefaultFullyConnectedFeedForwardLayerBuilderSession<C> extends
		DefaultDirectedNon3DLayerBuilderSession<FullyConnectedFeedForwardLayer, FullyConnectedFeedForwardLayerPropertiesBuilder<C>, FullyConnectedLayerAxonsConfig, FullyConnectedLayerConfigBuilder, FullyConnectedFeedForwardLayerPropertiesBuilder<C>>
		implements FullyConnectedFeedForwardLayerBuilderSession<C>, FullyConnectedFeedForwardLayerPropertiesBuilder<C> {

	private static WeightsMatrix DEFAULT_UNINITIALISED_WEIGHTS_MATRIX =  new WeightsMatrixImpl(null, 
			new WeightsFormatImpl(Arrays.asList(Dimension.INPUT_FEATURE), 
					Arrays.asList(Dimension.INPUT_FEATURE), WeightsMatrixOrientation.ROWS_SPAN_OUTPUT_DIMENSIONS));

	
	private Supplier<C> originalLayerContainer;
	private FullyConnectedLayerConfigBuilder layerConfigBuilder;

	public DefaultFullyConnectedFeedForwardLayerBuilderSession(String layerName,
			DirectedLayerFactory directedLayerFactory, Supplier<C> layerContainer,
			Consumer<FullyConnectedFeedForwardLayer> completedLayerConsumer) {
		super(layerName, directedLayerFactory, null, completedLayerConsumer);
		withLayerContainer(() -> this);
		this.originalLayerContainer = layerContainer;
		this.layerConfigBuilder = createConfigBuilder();
	}

	@Override
	protected FullyConnectedFeedForwardLayer build(FullyConnectedLayerAxonsConfig layerConfig) {
		
		WeightsMatrix weightsMatrix = layerConfigBuilder.getWeightsMatrix();
		BiasMatrix biasMatrix = layerConfigBuilder.getBiasMatrix();
		BatchNormConfig<Neurons> batchNormConfig = layerConfigBuilder.getBatchNormConfig();
		
		// If no weights matrix has been explicitly configured, create a weights with null matrix and default format.
		// If no bias matrix has been set, it will be defaulted by the directedlayerfactory if left neurons have bias unit.
		if (weightsMatrix == null) {
			weightsMatrix = DEFAULT_UNINITIALISED_WEIGHTS_MATRIX;
		}
		
		if (layerConfig.getActivationFunctionType() == null) {
			return directedLayerFactory.createFullyConnectedFeedForwardLayer(layerName, layerConfig, weightsMatrix, biasMatrix,
					ActivationFunctionType.getBaseType(ActivationFunctionBaseType.LINEAR),
					new ActivationFunctionProperties(), batchNormConfig);
		} else {
			return directedLayerFactory.createFullyConnectedFeedForwardLayer(layerName, layerConfig, weightsMatrix, biasMatrix,
					layerConfig.getActivationFunctionType(), layerConfig.getActivationFunctionProperties(), batchNormConfig);
		}
	}

	@Override
	public FullyConnectedFeedForwardLayerPropertiesBuilder<C> withInputNeurons(Neurons leftNeurons) {
		layerConfigBuilder.withInputNeurons(leftNeurons);
		return this;
	}

	@Override
	protected FullyConnectedFeedForwardLayerPropertiesBuilder<C> getPropertiesBuilderInstance() {
		return this;
	}

	@Override
	protected FullyConnectedLayerConfigBuilder createConfigBuilder() {
		return new FullyConnectedLayerConfigBuilder();
	}

	@Override
	public C withActivationFunction(DifferentiableActivationFunction activationFunction) {

		layerConfigBuilder.withActivationFunction(activationFunction);

		FullyConnectedLayerAxonsConfig axons3DConfig = layerConfigBuilder.build();

		FullyConnectedFeedForwardLayer layer = build(axons3DConfig);

		addCompletedLayer(layer);

		return originalLayerContainer.get();
	}

	@Override
	public C withActivationFunction(ActivationFunctionType activationFunctionType,
			ActivationFunctionProperties activationFunctionProperties) {

		layerConfigBuilder.withActivationFunctionType(activationFunctionType);
		if (activationFunctionProperties != null) {
			layerConfigBuilder.withActivationFunctionProperties(activationFunctionProperties);
		} else {
			layerConfigBuilder.withActivationFunctionProperties(new ActivationFunctionProperties());
		}

		FullyConnectedLayerAxonsConfig axons3DConfig = layerConfigBuilder.build();

		FullyConnectedFeedForwardLayer layer = build(axons3DConfig);

		addCompletedLayer(layer);

		return originalLayerContainer.get();
	}
	
	@Override
	public C withActivationFunction(ActivationFunctionType activationFunctionType) {
		return withActivationFunction(activationFunctionType, null);
	}


	@Override
	public C withActivationFunction(ActivationFunctionBaseType activationFunctionBaseType) {
		return withActivationFunction(ActivationFunctionType.getBaseType(activationFunctionBaseType), null);
	}


	@Override
	public C withActivationFunction(ActivationFunctionBaseType activationFunctionBaseType, ActivationFunctionProperties activationFunctionProperties) {
		return withActivationFunction(ActivationFunctionType.getBaseType(activationFunctionBaseType), activationFunctionProperties);
	}

	@Override
	public FullyConnectedFeedForwardLayerPropertiesBuilder<C> withOutputNeurons(Neurons outputNeurons) {

		layerConfigBuilder.withOutputNeurons(outputNeurons);

		return this;
	}

	@Override
	public FullyConnectedFeedForwardLayerPropertiesBuilder<C> withBiasMatrix(BiasMatrix biasMatrix) {

		layerConfigBuilder.withBiasMatrix(biasMatrix);
		return this;
	}

	@Override
	public FullyConnectedFeedForwardLayerPropertiesBuilder<C> withBiasUnit() {
		layerConfigBuilder.withBiasUnit();
		return this;
	}

	@Override
	public FullyConnectedFeedForwardLayerPropertiesBuilder<C> withWeightsFormat(WeightsFormat weightsFormat) {
		layerConfigBuilder.withWeightsFormat(weightsFormat);
		return this;
	}

	@Override
	public FullyConnectedFeedForwardLayerPropertiesBuilder<C> withWeightsMatrix(WeightsMatrix weightsMatrix) {
		layerConfigBuilder.withWeightsMatrix(weightsMatrix);
		return this;
	}

	@Override
	public FullyConnectedFeedForwardLayerPropertiesBuilder<C> withBatchNormConfig(
			Consumer<BatchNormConfig<Neurons>> batchNormConfigConfiger) {
		
		BatchNormConfig<Neurons> batchNormConfig = new BatchNormConfig<>(BatchNormDimension.INPUT_FEATURE);
		batchNormConfigConfiger.accept(batchNormConfig);
		layerConfigBuilder.withBatchNormConfig(batchNormConfig);
		return this;
	}
}
