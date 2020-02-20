package org.ml4j.nn.sessions;

import java.util.Arrays;
import java.util.function.Consumer;
import java.util.function.Supplier;

import org.ml4j.nn.activationfunctions.ActivationFunctionBaseType;
import org.ml4j.nn.activationfunctions.ActivationFunctionProperties;
import org.ml4j.nn.activationfunctions.ActivationFunctionType;
import org.ml4j.nn.activationfunctions.DifferentiableActivationFunction;
import org.ml4j.nn.axons.AxonsContextConfigurer;
import org.ml4j.nn.axons.BatchNormAxonsConfig;
import org.ml4j.nn.axons.BatchNormAxonsConfigConfigurer;
import org.ml4j.nn.axons.BatchNormConfig.BatchNormDimension;
import org.ml4j.nn.axons.BiasVector;
import org.ml4j.nn.axons.ConvolutionalAxonsConfig;
import org.ml4j.nn.axons.WeightsFormat;
import org.ml4j.nn.axons.WeightsFormatImpl;
import org.ml4j.nn.axons.WeightsMatrix;
import org.ml4j.nn.axons.WeightsMatrixImpl;
import org.ml4j.nn.axons.WeightsMatrixOrientation;
import org.ml4j.nn.layers.ConvolutionalFeedForwardLayer;
import org.ml4j.nn.layers.DirectedLayerFactory;
import org.ml4j.nn.layers.builders.ConvolutionalFeedForwardLayerPropertiesBuilder;
import org.ml4j.nn.layers.builders.ConvolutionalLayerAxonsConfig;
import org.ml4j.nn.layers.builders.ConvolutionalLayerConfigBuilder;
import org.ml4j.nn.neurons.Neurons3D;
import org.ml4j.nn.neurons.format.features.Dimension;

public class DefaultConvolutionalFeedForwardLayerBuilderSession<C> extends
		DefaultDirected3DLayerBuilderSession<ConvolutionalFeedForwardLayer, ConvolutionalFeedForwardLayerPropertiesBuilder<C>, ConvolutionalLayerAxonsConfig, ConvolutionalLayerConfigBuilder, ConvolutionalFeedForwardLayerPropertiesBuilder<C>>
		implements ConvolutionalFeedForwardLayerBuilderSession<C>, ConvolutionalFeedForwardLayerPropertiesBuilder<C> {

	private static WeightsMatrix DEFAULT_UNINITIALISED_WEIGHTS_MATRIX =  new WeightsMatrixImpl(null, 
			new WeightsFormatImpl(Arrays.asList(Dimension.INPUT_DEPTH, Dimension.FILTER_HEIGHT, Dimension.FILTER_WIDTH),
					Arrays.asList(Dimension.OUTPUT_DEPTH),
					WeightsMatrixOrientation.ROWS_SPAN_OUTPUT_DIMENSIONS));
	
	private Supplier<C> originalLayerContainer;
	private ConvolutionalLayerConfigBuilder layerConfigBuilder;

	public DefaultConvolutionalFeedForwardLayerBuilderSession(String layerName,
			DirectedLayerFactory directedLayerFactory, Supplier<C> layerContainer,
			Consumer<ConvolutionalFeedForwardLayer> completedLayerConsumer) {
		super(layerName, directedLayerFactory, null, completedLayerConsumer);
		withLayerContainer(() -> this);
		this.originalLayerContainer = layerContainer;
		this.layerConfigBuilder = createConfigBuilder();
	}
	

	@Override
	protected ConvolutionalFeedForwardLayer build(ConvolutionalLayerAxonsConfig layerConfig) {
		WeightsMatrix weightsMatrix = layerConfigBuilder.getWeightsMatrix();
		BiasVector biasMatrix = layerConfigBuilder.getBiasVector();
		BatchNormAxonsConfig<Neurons3D> batchNormAxonsConfig = layerConfigBuilder.getBatchNormAxonsConfig();
		
	
		// If no weights matrix has been explicitly configured, create a weights with null matrix and default format.
		// If no bias matrix has been set, it will be defaulted by the directedlayerfactory if left neurons have bias unit.
		if (weightsMatrix == null) {
			weightsMatrix = DEFAULT_UNINITIALISED_WEIGHTS_MATRIX;
		}
		
		if (batchNormAxonsConfig != null) {
		
			if (batchNormAxonsConfig.getNeurons() == null) {
				batchNormAxonsConfig.withNeurons(layerConfigBuilder.getRightNeurons());
			} else {
				if (!layerConfigBuilder.getRightNeurons().equals(batchNormAxonsConfig.getNeurons())) {
					throw new IllegalStateException("Neurons set on BatchNormAxonsConfig should match the output neurons of "
							+ "the ConvolutionalAxons");
				}
			}
		
		}
		
		ConvolutionalAxonsConfig convolutionalAxonsConfig = ConvolutionalAxonsConfig.create(layerConfig);
		
		if (layerConfigBuilder.getAxonsContextConfigurer() != null) {
			convolutionalAxonsConfig = convolutionalAxonsConfig.withAxonsContextConfigurer(layerConfigBuilder.getAxonsContextConfigurer());
		}
				
		ConvolutionalFeedForwardLayer layer;
		if (layerConfig.getActivationFunctionType() == null) {
			layer = directedLayerFactory.createConvolutionalFeedForwardLayer(layerName, convolutionalAxonsConfig, weightsMatrix, biasMatrix,
					ActivationFunctionType.getBaseType(ActivationFunctionBaseType.LINEAR),
					new ActivationFunctionProperties(), batchNormAxonsConfig);
		} else {
			layer = directedLayerFactory.createConvolutionalFeedForwardLayer(layerName, convolutionalAxonsConfig, weightsMatrix, biasMatrix,
					layerConfig.getActivationFunctionType(), layerConfig.getActivationFunctionProperties(), batchNormAxonsConfig);
		}
		
		return layer;
		
	}


	@Override
	public C withActivationFunction(DifferentiableActivationFunction activationFunction) {
		layerConfigBuilder.withActivationFunction(activationFunction);

		ConvolutionalLayerAxonsConfig axons3DConfig = layerConfigBuilder.build(configPopulator);

		ConvolutionalFeedForwardLayer layer = build(axons3DConfig);

		addCompletedLayer(layer);

		return originalLayerContainer.get();
	}
	
	

	@Override
	public ConvolutionalFeedForwardLayerPropertiesBuilder<C> withOutputNeurons(Neurons3D outputNeurons) {
		this.layerConfigBuilder.withOutputNeurons(outputNeurons);
		return this;
	}
	

	@Override
	public ConvolutionalFeedForwardLayerPropertiesBuilder<C> withConfig(
			Consumer<ConvolutionalLayerConfigBuilder> configConfigurer) {
		
		configConfigurer.accept(layerConfigBuilder);
			
		return this;
	}


	@Override
	public C withActivationFunction(ActivationFunctionType activationFunctionType, ActivationFunctionProperties activationFunctionProperties) {
		layerConfigBuilder.withActivationFunctionType(activationFunctionType);

		layerConfigBuilder.withActivationFunctionType(activationFunctionType);
		if (activationFunctionProperties != null) {
			layerConfigBuilder.withActivationFunctionProperties(activationFunctionProperties);
		} else {
			layerConfigBuilder.withActivationFunctionProperties(new ActivationFunctionProperties());
		}
		
		ConvolutionalLayerAxonsConfig axons3DConfig = layerConfigBuilder.build(configPopulator);

		ConvolutionalFeedForwardLayer layer = build(axons3DConfig);

		addCompletedLayer(layer);

		return originalLayerContainer.get();
	}

	@Override
	public ConvolutionalFeedForwardLayerPropertiesBuilder<C> withBatchNormAxonsConfig(
			BatchNormAxonsConfigConfigurer<Neurons3D> batchNormConfigConfigurer) {
	
		BatchNormAxonsConfig<Neurons3D> batchNormAxonsConfig = BatchNormAxonsConfig.create(BatchNormDimension.CHANNEL);
		batchNormConfigConfigurer.accept(batchNormAxonsConfig);
		layerConfigBuilder.withBatchNormAxonsConfig(batchNormAxonsConfig);
		return this;
	}

	@Override
	public ConvolutionalFeedForwardLayerPropertiesBuilder<C> withBiasVector(BiasVector biasMatrix) {
		layerConfigBuilder.withBiasVector(biasMatrix);
		return this;
	}

	@Override
	public ConvolutionalFeedForwardLayerPropertiesBuilder<C> withBiasUnit() {
		layerConfigBuilder.withBiasUnit();
		return this;
	}

	@Override
	public ConvolutionalFeedForwardLayerPropertiesBuilder<C> withWeightsFormat(WeightsFormat weightsFormat) {
		layerConfigBuilder.withWeightsFormat(weightsFormat);
		return this;
	}

	@Override
	public ConvolutionalFeedForwardLayerPropertiesBuilder<C> withWeightsMatrix(WeightsMatrix weightsMatrix) {
		layerConfigBuilder.withWeightsMatrix(weightsMatrix);
		return this;
	}

	@Override
	protected ConvolutionalLayerConfigBuilder createConfigBuilder() {
		return this.leftNeurons == null ? new ConvolutionalLayerConfigBuilder() : new ConvolutionalLayerConfigBuilder(leftNeurons);
	}

	@Override
	public ConvolutionalFeedForwardLayerPropertiesBuilder<C> withInputNeurons(Neurons3D inputNeurons) {
		layerConfigBuilder.withInputNeurons(inputNeurons);
		this.leftNeurons = inputNeurons;
		return this;
	}
	
	@Override
	protected ConvolutionalFeedForwardLayerPropertiesBuilder<C> getPropertiesBuilderInstance() {
		return this;
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
	public ConvolutionalFeedForwardLayerPropertiesBuilder<C> withAxonsContextConfigurer(AxonsContextConfigurer axonsContextConfigurer) {
		layerConfigBuilder.withAxonsContextConfigurer(axonsContextConfigurer);
		return this;
	}
	
}
