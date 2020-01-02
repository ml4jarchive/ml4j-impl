/*
 * Copyright 2017 the original author or authors.
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

package org.ml4j.nn;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Optional;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Future;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.Consumer;
import java.util.function.Supplier;
import java.util.stream.Stream;

import org.ml4j.InterrimMatrix;
import org.ml4j.Matrix;
import org.ml4j.nn.activationfunctions.ActivationFunctionType;
import org.ml4j.nn.activationfunctions.DifferentiableActivationFunction;
import org.ml4j.nn.axons.AxonsGradient;
import org.ml4j.nn.axons.AxonWeightsAdjustment;
import org.ml4j.nn.axons.AxonWeightsAdjustmentDirection;
import org.ml4j.nn.axons.AxonWeightsAdjustmentImpl;
import org.ml4j.nn.axons.TrainableAxons;
import org.ml4j.nn.components.DirectedComponentType;
import org.ml4j.nn.components.factories.DirectedComponentFactory;
import org.ml4j.nn.components.generic.DirectedComponentChain;
import org.ml4j.nn.components.onetone.DefaultChainableDirectedComponent;
import org.ml4j.nn.components.onetone.TrailingActivationFunctionDirectedComponentChain;
import org.ml4j.nn.components.onetone.TrailingActivationFunctionDirectedComponentChainActivation;
import org.ml4j.nn.costfunctions.CostFunction;
import org.ml4j.nn.costfunctions.CostFunctionGradient;
import org.ml4j.nn.costfunctions.CrossEntropyCostFunction;
import org.ml4j.nn.costfunctions.DeltaRuleCostFunctionGradientImpl;
import org.ml4j.nn.costfunctions.MultiClassCrossEntropyCostFunction;
import org.ml4j.nn.costfunctions.SumSquaredErrorCostFunction;
import org.ml4j.nn.datasets.LabeledData;
import org.ml4j.nn.datasets.LabeledDataSet;
import org.ml4j.nn.neurons.NeuronsActivation;
import org.ml4j.nn.neurons.NeuronsActivationFeatureOrientation;
import org.ml4j.nn.neurons.NeuronsActivationImpl;
import org.ml4j.nn.optimisation.GradientDescentOptimisationStrategy;
import org.ml4j.nn.optimisation.TrainingLearningRateAdjustmentStrategy;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.ml4j.nn.components.onetoone.TrailingActivationFunctionDirectedComponentChainImpl;

/**
 * Default base implementation of a FeedForwardNeuralNetwork.
 *
 * @author Michael Lavelle
 */
public abstract class FeedForwardNeuralNetworkBase<C extends FeedForwardNeuralNetworkContext, H extends DirectedComponentChain<NeuronsActivation, ? extends DefaultChainableDirectedComponent<?, ?>, ?, ?>, N extends FeedForwardNeuralNetwork<C, N>>
		implements FeedForwardNeuralNetwork<C, N> {

	private static final Logger LOGGER = LoggerFactory.getLogger(FeedForwardNeuralNetworkBase.class);

	protected H initialisingComponentChain;
	protected TrailingActivationFunctionDirectedComponentChain trailingActivationFunctionComponentChain;
	
	protected GradientAccumulator gradientAccumulator;

	private C lastEpochTrainingContext;

	/**
	 * Default serialization id.
	 */
	private static final long serialVersionUID = 1L;

	/**
	 * Constructor for a multi-layer FeedForwardNeuralNetwork.
	 * 
	 * @param layers The layers
	 */
	public FeedForwardNeuralNetworkBase(DirectedComponentFactory directedComponentFactory, H initialisingComponentChain) {
		this.initialisingComponentChain = initialisingComponentChain;
		this.trailingActivationFunctionComponentChain = new TrailingActivationFunctionDirectedComponentChainImpl(directedComponentFactory,
				initialisingComponentChain.getComponents());
		this.gradientAccumulator = new LocalGradientAccumulator();
	}
	
	/**
	 * Constructor for a multi-layer FeedForwardNeuralNetwork.
	 * 
	 * @param layers The layers
	 */
	protected FeedForwardNeuralNetworkBase(H initialisingComponentChain, TrailingActivationFunctionDirectedComponentChain trailingActivationFunctionComponentChain) {
		this.initialisingComponentChain = initialisingComponentChain;
		this.trailingActivationFunctionComponentChain = trailingActivationFunctionComponentChain;
		this.gradientAccumulator = new LocalGradientAccumulator();
	}

	// TODO - Refactor these methods
	protected void train(NeuronsActivation trainingDataActivations, NeuronsActivation trainingLabelActivations,
			C trainingContext) {

		if (trainingDataActivations
				.getFeatureOrientation() != NeuronsActivationFeatureOrientation.ROWS_SPAN_FEATURE_SET) {
			throw new IllegalArgumentException(
					"Only neurons actiavation with ROWS_SPAN_FEATURE_SET " + "orientation supported currently");
		}

		if (trainingLabelActivations
				.getFeatureOrientation() != NeuronsActivationFeatureOrientation.ROWS_SPAN_FEATURE_SET) {
			throw new IllegalArgumentException(
					"Only neurons actiavation with ROWS_SPAN_FEATURE_SET " + "orientation supported currently");
		}

		final int numberOfEpochs = trainingContext.getTrainingEpochs();

		LOGGER.info("Training the FeedForwardNeuralNetwork for " + numberOfEpochs + " epochs");

		CostAndGradients costAndGradients = null;

		int iterationIndex = 0;

		int epochStartIndex = (lastEpochTrainingContext == null
				|| lastEpochTrainingContext.getLastTrainingEpochIndex() == null) ? 0
						: (lastEpochTrainingContext.getLastTrainingEpochIndex() + 1);

		for (int epochIndex = epochStartIndex; epochIndex < epochStartIndex + numberOfEpochs; epochIndex++) {

			if (trainingContext.getTrainingMiniBatchSize() == null) {
				costAndGradients = getCostAndGradients(trainingDataActivations, trainingLabelActivations,
						trainingContext);

				LOGGER.info("Epoch:" + epochIndex + " Cost:" + costAndGradients.getAverageCost());
				// Timings.printTimings();
				int batchIndex = epochIndex;

				List<AxonsGradient> averageTrainableAxonsGradients = costAndGradients
						.getAverageTrainableAxonsGradients();

				adjustConnectionWeights(trainingContext, averageTrainableAxonsGradients, epochIndex, batchIndex,
						iterationIndex);

				for (AxonsGradient axonsGradient : averageTrainableAxonsGradients) {
					axonsGradient.getWeightsGradient().close();
					if (axonsGradient.getLeftToRightBiasGradient() != null) {
						axonsGradient.getLeftToRightBiasGradient().close();
					}
					if (axonsGradient.getRightToLeftBiasGradient() != null) {
						axonsGradient.getRightToLeftBiasGradient().close();
					}
				}

				iterationIndex++;
			} else {
				int miniBatchSize = trainingContext.getTrainingMiniBatchSize();
				Matrix activations = trainingDataActivations.getActivations(trainingContext.getMatrixFactory());
				int numberOfTrainingElements = trainingDataActivations.getExampleCount();
				int numberOfBatches = (numberOfTrainingElements - 1) / miniBatchSize + 1;
				for (int batchIndex = 0; batchIndex < numberOfBatches; batchIndex++) {
					int startColumnIndex = batchIndex * miniBatchSize;
					int endColumnIndex = Math.min(startColumnIndex + miniBatchSize - 1, numberOfTrainingElements - 1);
					int[] columnsIndexes = new int[endColumnIndex - startColumnIndex + 1];
					for (int c = startColumnIndex; c <= endColumnIndex; c++) {
						columnsIndexes[c - startColumnIndex] = c;
					}
					try (InterrimMatrix dataBatch = activations.getColumns(columnsIndexes).asInterrimMatrix();
							InterrimMatrix labelBatch = trainingLabelActivations
									.getActivations(trainingContext.getMatrixFactory()).getColumns(columnsIndexes)
									.asInterrimMatrix()) {

						NeuronsActivation batchDataActivations = new NeuronsActivationImpl(dataBatch,
								trainingDataActivations.getFeatureOrientation());

						NeuronsActivation batchLabelActivations = new NeuronsActivationImpl(labelBatch,
								trainingLabelActivations.getFeatureOrientation());

						costAndGradients = getCostAndGradients(batchDataActivations, batchLabelActivations,
								trainingContext);

						LOGGER.debug("Epoch:" + epochIndex + " batch " + batchIndex + " Cost:"
								+ costAndGradients.getAverageCost());
						// Timings.printTimings();

						List<AxonsGradient> averageTrainableAxonsGradients = costAndGradients
								.getAverageTrainableAxonsGradients();

						adjustConnectionWeights(trainingContext, averageTrainableAxonsGradients, epochIndex, batchIndex,
								iterationIndex);

						for (AxonsGradient axonsGradient : averageTrainableAxonsGradients) {
							axonsGradient.getWeightsGradient().close();
							if (axonsGradient.getLeftToRightBiasGradient() != null) {
								axonsGradient.getLeftToRightBiasGradient().close();
							}
							if (axonsGradient.getRightToLeftBiasGradient() != null) {
								axonsGradient.getRightToLeftBiasGradient().close();
							}
						}
					}
					iterationIndex++;
				}

				LOGGER.info("Epoch:" + epochIndex + " Cost:" + costAndGradients.getAverageCost());
				// Timings.printTimings();

				lastEpochTrainingContext = trainingContext;
			}
		}
	}

	protected void train(LabeledDataSet<NeuronsActivation, NeuronsActivation> trainingDataSet, C trainingContext) {
		

		final int numberOfEpochs = trainingContext.getTrainingEpochs();

		LOGGER.info("Training the FeedForwardNeuralNetwork for " + numberOfEpochs + " epochs");

		List<CostAndGradients> costAndGradientsList = new ArrayList<>();

		AtomicInteger iterationIndex = new AtomicInteger(0);

		int epochStartIndex = (lastEpochTrainingContext == null
				|| lastEpochTrainingContext.getLastTrainingEpochIndex() == null) ? 0
						: (lastEpochTrainingContext.getLastTrainingEpochIndex() + 1);

		AtomicInteger batchIndex = new AtomicInteger(0);

		for (int epochIndex = epochStartIndex; epochIndex < epochStartIndex + numberOfEpochs; epochIndex++) {

			final int epochIndex2 = epochIndex;

			try (Stream<LabeledData<NeuronsActivation, NeuronsActivation>> trainingDataStream = trainingDataSet
					.stream()) {

				if (trainingContext.getTrainingMiniBatchSize() == null) {

					trainingDataStream.forEach(batch -> {

						NeuronsActivation batchDataActivations = batch.getData();

						NeuronsActivation batchLabelActivations = batch.getLabel();
						
						if (batchDataActivations
								.getFeatureOrientation() != NeuronsActivationFeatureOrientation.ROWS_SPAN_FEATURE_SET) {
							throw new IllegalArgumentException("Only neurons actiavation with ROWS_SPAN_FEATURE_SET "
									+ "orientation supported currently");
						}

						if (batchLabelActivations
								.getFeatureOrientation() != NeuronsActivationFeatureOrientation.ROWS_SPAN_FEATURE_SET) {
							throw new IllegalArgumentException("Only neurons actiavation with ROWS_SPAN_FEATURE_SET "
									+ "orientation supported currently");
						}

						CostAndGradients costAndGradients = getCostAndGradients(batchDataActivations,
								batchLabelActivations, trainingContext);

						costAndGradientsList.add(costAndGradients);

						LOGGER.debug("Epoch:" + epochIndex2 + " batch " + batchIndex + " Cost:"
								+ costAndGradients.getAverageCost());
						// Timings.printTimings();

						List<AxonsGradient> averageTrainableAxonsGradients = costAndGradients
								.getAverageTrainableAxonsGradients();

						adjustConnectionWeights(trainingContext, averageTrainableAxonsGradients, epochIndex2,
								batchIndex.get(), iterationIndex.get());

						for (AxonsGradient axonsGradient : averageTrainableAxonsGradients) {
							axonsGradient.getWeightsGradient().close();
							if (axonsGradient.getLeftToRightBiasGradient() != null) {
								axonsGradient.getLeftToRightBiasGradient().close();
							}
							if (axonsGradient.getRightToLeftBiasGradient() != null) {
								axonsGradient.getRightToLeftBiasGradient().close();
							}
						}

						iterationIndex.addAndGet(1);
						batchIndex.addAndGet(1);

					});

				}

				LOGGER.info("Epoch:" + epochIndex + " Cost:"
						+ costAndGradientsList.get(costAndGradientsList.size() - 1).getAverageCost());

				lastEpochTrainingContext = trainingContext;
			}
		}
	}

	protected void train(Supplier<Stream<LabeledData<NeuronsActivation, NeuronsActivation>>> trainingDataSet, C trainingContext,
			Consumer<Float> epochAverageCostHandler) {

		final int numberOfEpochs = trainingContext.getTrainingEpochs();

		LOGGER.info("Training the FeedForwardNeuralNetwork for " + numberOfEpochs + " epochs");

		List<CostAndGradients> costAndGradientsList = new ArrayList<>();

		AtomicInteger iterationIndex = new AtomicInteger(0);

		int epochStartIndex = (lastEpochTrainingContext == null
				|| lastEpochTrainingContext.getLastTrainingEpochIndex() == null) ? 0
						: (lastEpochTrainingContext.getLastTrainingEpochIndex() + 1);

		AtomicInteger batchIndex = new AtomicInteger(0);

		for (int epochIndex = epochStartIndex; epochIndex < epochStartIndex + numberOfEpochs; epochIndex++) {

			final int epochIndex2 = epochIndex;

			try (Stream<LabeledData<NeuronsActivation, NeuronsActivation>> trainingDataStream = 
					trainingDataSet.get();
				) {
				

			if (trainingContext.getTrainingMiniBatchSize() == null) {

				
				trainingDataStream.forEach(batch -> {

					NeuronsActivation batchDataActivations = batch.getData();

					NeuronsActivation batchLabelActivations = batch.getLabel();
					
					if (batchDataActivations
							.getFeatureOrientation() != NeuronsActivationFeatureOrientation.ROWS_SPAN_FEATURE_SET) {
						throw new IllegalArgumentException("Only neurons actiavation with ROWS_SPAN_FEATURE_SET "
								+ "orientation supported currently");
					}

					if (batchLabelActivations
							.getFeatureOrientation() != NeuronsActivationFeatureOrientation.ROWS_SPAN_FEATURE_SET) {
						throw new IllegalArgumentException("Only neurons actiavation with ROWS_SPAN_FEATURE_SET "
								+ "orientation supported currently");
					}

					CostAndGradients costAndGradients = getCostAndGradients(batchDataActivations, batchLabelActivations,
							trainingContext);
					
					costAndGradientsList.add(costAndGradients);
					

					LOGGER.info("Epoch:" + epochIndex2 + " batch " + batchIndex + " Cost:"
							+ costAndGradients.getAverageCost());
					
					Optional<Future<List<AxonsGradient>>> averageAxonsGradientsResult = gradientAccumulator.submitCostAndGradients(costAndGradients);
		
					if (averageAxonsGradientsResult.isPresent()) {
	
						List<AxonsGradient> averageTrainableAxonsGradients;
						try {
							averageTrainableAxonsGradients = averageAxonsGradientsResult.get().get();
						
							adjustConnectionWeights(trainingContext, averageTrainableAxonsGradients, epochIndex2,
									batchIndex.get(), iterationIndex.get());
		
							for (AxonsGradient axonsGradient : averageTrainableAxonsGradients) {
								axonsGradient.getWeightsGradient().close();
								if (axonsGradient.getLeftToRightBiasGradient() != null) {
									axonsGradient.getLeftToRightBiasGradient().close();
								}
								if (axonsGradient.getRightToLeftBiasGradient() != null) {
									axonsGradient.getRightToLeftBiasGradient().close();
								}
							}
						
						} catch (InterruptedException e) {
							LOGGER.error("Interrupted when waiting for response from gradient accumulator", e);
						} catch (ExecutionException e) {
							LOGGER.error("Execution exception when waiting for response from gradient accumulator", e);
						}
					}
					

					iterationIndex.addAndGet(1);
					batchIndex.addAndGet(1);
				});
				
			}
				CostAndGradients costAndGradients = costAndGradientsList.get(costAndGradientsList.size() - 1);
				costAndGradientsList.clear();
				epochAverageCostHandler
						.accept(costAndGradients.getAverageCost());
				LOGGER.debug("Epoch:" + epochIndex + " Cost:"
						+ costAndGradients.getAverageCost());
				lastEpochTrainingContext = trainingContext;
			}
		}
	}

	protected void train(Stream<LabeledData<NeuronsActivation, NeuronsActivation>> trainingDataActivations,
			C trainingContext) {

		int iterationIndex = 0;

		int epochStartIndex = (lastEpochTrainingContext == null
				|| lastEpochTrainingContext.getLastTrainingEpochIndex() == null) ? 0
						: (lastEpochTrainingContext.getLastTrainingEpochIndex() + 1);

		int epochIndex = epochStartIndex;

		trainingDataActivations.forEach(labeledData -> {

			if (trainingContext.getTrainingMiniBatchSize() == null) {
				CostAndGradients costAndGradients = getCostAndGradients(labeledData.getData(), labeledData.getLabel(),
						trainingContext);

				LOGGER.info("Epoch:" + epochIndex + " Cost:" + costAndGradients.getAverageCost());
				int batchIndex = epochIndex;

				List<AxonsGradient> averageTrainableAxonsGradients = costAndGradients
						.getAverageTrainableAxonsGradients();

				adjustConnectionWeights(trainingContext, averageTrainableAxonsGradients, epochIndex, batchIndex,
						iterationIndex);

				for (AxonsGradient axonsGradient : averageTrainableAxonsGradients) {
					axonsGradient.getWeightsGradient().close();
					if (axonsGradient.getLeftToRightBiasGradient() != null) {
						axonsGradient.getLeftToRightBiasGradient().close();
					}
					if (axonsGradient.getRightToLeftBiasGradient() != null) {
						axonsGradient.getRightToLeftBiasGradient().close();
					}
				}

			} else {
				throw new IllegalArgumentException("Training batch mini batch size not supported");
			}

			lastEpochTrainingContext = trainingContext;

		});
	}

	protected CostAndGradientsImpl getCostAndGradients(NeuronsActivation inputActivations,
			NeuronsActivation desiredOutputActivations, C trainingContext) {

		int numberOfTrainingExamples = inputActivations.getColumns();

		if (inputActivations.getFeatureOrientation() != NeuronsActivationFeatureOrientation.ROWS_SPAN_FEATURE_SET) {
			throw new IllegalArgumentException(
					"Only neurons actiavation with ROWS_SPAN_FEATURE_SET " + "orientation supported currently");
		}

		if (desiredOutputActivations
				.getFeatureOrientation() != NeuronsActivationFeatureOrientation.ROWS_SPAN_FEATURE_SET) {
			throw new IllegalArgumentException(
					"Only neurons actiavation with ROWS_SPAN_FEATURE_SET " + "orientation supported currently");
		}

		final CostFunction costFunction = getCostFunction();

		// Forward propagate the trainingDataActivations through the entire Network
		ForwardPropagation forwardPropagation = forwardPropagate(inputActivations, trainingContext);

		// Obtain the cost from the cost function
		LOGGER.debug("Calculating total cost function cost");

		float totalCost = costFunction.getTotalCost(
				desiredOutputActivations.getActivations(trainingContext.getMatrixFactory()),
				forwardPropagation.getOutput().getActivations(trainingContext.getMatrixFactory()));

		float totalRegularisationCost = forwardPropagation.getTotalRegularisationCost(trainingContext);

		float totalCostWithRegularisation = totalCost + totalRegularisationCost;

		CostFunctionGradient costFunctionGradient = new DeltaRuleCostFunctionGradientImpl(trainingContext.getMatrixFactory(), costFunction,
				desiredOutputActivations, forwardPropagation.getOutput());

		// Back propagate the cost function gradient through the network
		BackPropagation backPropagation = forwardPropagation.backPropagate(costFunctionGradient, trainingContext);
	

		// Obtain the gradients of each set of Axons we wish to train - for this example
		// it is
		// all the Axons
		List<Supplier<AxonsGradient>> totalTrainableAxonsGradientSuppliers = backPropagation.getGradient()
				.getTotalTrainableAxonsGradients();

		Collections.reverse(totalTrainableAxonsGradientSuppliers);

		LOGGER.debug("Calculating gradients");
		List<AxonsGradient> totalTrainableAxonsGradients = new ArrayList<AxonsGradient>();
		for (Supplier<AxonsGradient> totalTrainableAxonsGradientSupplier : totalTrainableAxonsGradientSuppliers) {
			AxonsGradient gradient = totalTrainableAxonsGradientSupplier.get();
			if (gradient != null) {
				totalTrainableAxonsGradients.add(gradient);
			}
		}

		LOGGER.debug("Gradient count:" + totalTrainableAxonsGradients.size());
		
		return new CostAndGradientsImpl(totalCostWithRegularisation, totalTrainableAxonsGradients,
				numberOfTrainingExamples);

	}

	protected float getTrainingLearningRate(C trainingContext, int epochIndex, int batchIndex, int iterationIndex) {

		TrainingLearningRateAdjustmentStrategy adjustmentStrategy = trainingContext
				.getTrainingLearningRateAdjustmentStrategy();

		if (adjustmentStrategy != null) {
			return adjustmentStrategy.getTrainingLearningRate(trainingContext, epochIndex, batchIndex, iterationIndex);
		}

		return trainingContext.getTrainingLearningRate();
	}

	protected AxonsGradient getAdjustedAxonsGradient(AxonsGradient axonsGradient, int axonsIndex, C trainingContext,
			int epochIndex, int batchIndex, int iterationIndex) {

		GradientDescentOptimisationStrategy optimisationStrategy = trainingContext
				.getGradientDescentOptimisationStrategy();

		if (optimisationStrategy != null) {
			return optimisationStrategy.getAdjustedAxonsGradient(axonsGradient, axonsIndex, trainingContext, epochIndex,
					batchIndex, iterationIndex);
		}
		return axonsGradient;
	}

	private void adjustConnectionWeights(C trainingContext, List<AxonsGradient> trainableAxonsGradients, int epochIndex,
			int batchIndex, int iterationIndex) {
		int axonsIndex = 0;
		for (AxonsGradient axonsGradient : trainableAxonsGradients) {
			TrainableAxons<?, ?, ?> trainableAxons = axonsGradient.getAxons();
			// Transpose the axon gradients into matrices that correspond to the orientation
			// of the
			// connection weights ( COLUMNS_SPAN_FEATURE_SET )
			AxonsGradient adjustedAxonsGradient = getAdjustedAxonsGradient(axonsGradient, axonsIndex, trainingContext,
					epochIndex, batchIndex, iterationIndex);
			// Adjust the weights of each set of Axons by subtracting the learning-rate
			// scaled
			// gradient matrices
			Matrix weightsGradient = adjustedAxonsGradient.getWeightsGradient();
			
			try (InterrimMatrix weightsAdjustment = weightsGradient
					.mul((float) getTrainingLearningRate(trainingContext, epochIndex, batchIndex, iterationIndex))
					.asInterrimMatrix()) {
				AxonWeightsAdjustment axonWeightsAdjustment = null;
				if (trainableAxons.getLeftNeurons().hasBiasUnit()) {
					try (InterrimMatrix biasAdjustment = adjustedAxonsGradient.getLeftToRightBiasGradient()
							.mul(getTrainingLearningRate(trainingContext, epochIndex, batchIndex, iterationIndex))
							.asInterrimMatrix()) {
						axonWeightsAdjustment = new AxonWeightsAdjustmentImpl(weightsAdjustment, biasAdjustment);
						trainableAxons.adjustAxonWeights(axonWeightsAdjustment,
								AxonWeightsAdjustmentDirection.SUBTRACTION);
					}
				} else {
					axonWeightsAdjustment = new AxonWeightsAdjustmentImpl(weightsAdjustment);
					trainableAxons.adjustAxonWeights(axonWeightsAdjustment,
							AxonWeightsAdjustmentDirection.SUBTRACTION);
				}
			
			}
				

			
			axonsIndex++;
		}
	}

	@Override
	public ForwardPropagation forwardPropagate(NeuronsActivation inputActivation,
			FeedForwardNeuralNetworkContext context) {

		// int endLayerIndex = context.getEndLayerIndex() == null ? (getNumberOfLayers()
		// - 1) : context.getEndLayerIndex();

		LOGGER.debug("Forward propagating through FeedForwardNeuralNetwork");

		// Forward propagate through the layers
		TrailingActivationFunctionDirectedComponentChainActivation activation = trailingActivationFunctionComponentChain
				.forwardPropagate(inputActivation, context.getDirectedComponentsContext());

		// Construct a forward propagation
		ForwardPropagation forwardPropagation = new ForwardPropagationImpl(activation);

		if (context.getForwardPropagationListener() != null) {
			context.getForwardPropagationListener().onForwardPropagation(forwardPropagation);
		}

		return forwardPropagation;
	}

	@Override
	public Stream<ForwardPropagation> forwardPropagate(Stream<NeuronsActivation> inputActivation,
			C context) {

		// int endLayerIndex = context.getEndLayerIndex() == null ? (getNumberOfLayers()
		// - 1) : context.getEndLayerIndex();

		return inputActivation.map(act -> {

			LOGGER.debug("Forward propagating through FeedForwardNeuralNetwork");

			// Forward propagate through the layers
			TrailingActivationFunctionDirectedComponentChainActivation activation = trailingActivationFunctionComponentChain
					.forwardPropagate(act, context.getDirectedComponentsContext());

			// Construct a forward propagation
			ForwardPropagation forwardPropagation = new ForwardPropagationImpl(activation);

			if (context.getForwardPropagationListener() != null) {
				context.getForwardPropagationListener().onForwardPropagation(forwardPropagation);
			}

			return forwardPropagation;

		});
	}

	/**
	 * @return The default cost function for use by this Network.
	 */
	protected CostFunction getCostFunction() {

		DifferentiableActivationFunction activationFunction = trailingActivationFunctionComponentChain
				.getFinalComponent().getActivationFunction();

		if (activationFunction == null) {
			throw new UnsupportedOperationException(
					"Default cost function not yet defined for null activation function");
		}
		if (activationFunction.getActivationFunctionType() == ActivationFunctionType.SIGMOID) {
			LOGGER.debug("Defaulting to use CrossEntropyCostFunction");
			return new CrossEntropyCostFunction();
		} else if (activationFunction.getActivationFunctionType() == ActivationFunctionType.SOFTMAX) {
			LOGGER.debug("Defaulting to use MultiClassCrossEntropyCostFunction");
			return new MultiClassCrossEntropyCostFunction();
		} else if (activationFunction.getActivationFunctionType() == ActivationFunctionType.LINEAR) {
			LOGGER.debug("Defaulting to use SumSquredErrorCostFunction");
			return new SumSquaredErrorCostFunction();
		} else {
			throw new UnsupportedOperationException(
					"Default cost function not yet defined for:" + activationFunction.getClass());
		}
	}

	@Override
	public C getLastEpochTrainingContext() {
		return lastEpochTrainingContext;
	}
	
	@Override
	public DirectedComponentType getComponentType() {
		return DirectedComponentType.NETWORK;
	}	

}
