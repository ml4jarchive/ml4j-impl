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
import java.util.function.Supplier;

import org.ml4j.InterrimMatrix;
import org.ml4j.Matrix;
import org.ml4j.nn.activationfunctions.DifferentiableActivationFunction;
import org.ml4j.nn.activationfunctions.LinearActivationFunction;
import org.ml4j.nn.activationfunctions.SigmoidActivationFunction;
import org.ml4j.nn.activationfunctions.SoftmaxActivationFunction;
import org.ml4j.nn.axons.AxonsGradient;
import org.ml4j.nn.axons.ConnectionWeightsAdjustmentDirection;
import org.ml4j.nn.axons.TrainableAxons;
import org.ml4j.nn.components.DirectedComponentChain;
import org.ml4j.nn.components.TrailingActivationFunctionDirectedComponentChain;
import org.ml4j.nn.components.TrailingActivationFunctionDirectedComponentChainActivation;
import org.ml4j.nn.components.TrailingActivationFunctionDirectedComponentChainImpl;
import org.ml4j.nn.costfunctions.CostFunction;
import org.ml4j.nn.costfunctions.CostFunctionGradient;
import org.ml4j.nn.costfunctions.CrossEntropyCostFunction;
import org.ml4j.nn.costfunctions.DeltaRuleCostFunctionGradientImpl;
import org.ml4j.nn.costfunctions.MultiClassCrossEntropyCostFunction;
import org.ml4j.nn.costfunctions.SumSquaredErrorCostFunction;
import org.ml4j.nn.neurons.NeuronsActivation;
import org.ml4j.nn.neurons.NeuronsActivationFeatureOrientation;
import org.ml4j.nn.neurons.NeuronsActivationImpl;
import org.ml4j.nn.optimisation.GradientDescentOptimisationStrategy;
import org.ml4j.nn.optimisation.TrainingLearningRateAdjustmentStrategy;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Default base implementation of a FeedForwardNeuralNetwork.
 *
 * @author Michael Lavelle
 */
public abstract class FeedForwardNeuralNetworkBase<C extends FeedForwardNeuralNetworkContext, H extends DirectedComponentChain<NeuronsActivation, ?, ?, ?>,
    N extends FeedForwardNeuralNetwork<C,N>> 
    implements FeedForwardNeuralNetwork<C, N> {

  private static final Logger LOGGER = 
      LoggerFactory.getLogger(FeedForwardNeuralNetworkBase.class);
  
  protected H initialisingComponentChain;
  private TrailingActivationFunctionDirectedComponentChain<?> trailingActivationFunctionComponentChain;
  
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
  public FeedForwardNeuralNetworkBase(H initialisingComponentChain) {
	this.initialisingComponentChain = initialisingComponentChain;
    this.trailingActivationFunctionComponentChain = new TrailingActivationFunctionDirectedComponentChainImpl(initialisingComponentChain.getComponents());
  }
 
  protected void train(NeuronsActivation trainingDataActivations,
      NeuronsActivation trainingLabelActivations, C trainingContext) {

	  if (trainingDataActivations
		        .getFeatureOrientation() != NeuronsActivationFeatureOrientation.ROWS_SPAN_FEATURE_SET) {
		      throw new IllegalArgumentException("Only neurons actiavation with ROWS_SPAN_FEATURE_SET "
		          + "orientation supported currently");
		    }
	  
	  if (trainingLabelActivations
		        .getFeatureOrientation() != NeuronsActivationFeatureOrientation.ROWS_SPAN_FEATURE_SET) {
		      throw new IllegalArgumentException("Only neurons actiavation with ROWS_SPAN_FEATURE_SET "
		          + "orientation supported currently");
		    }  
	  

    final int numberOfEpochs = trainingContext.getTrainingEpochs();

    LOGGER.info("Training the FeedForwardNeuralNetwork for " + numberOfEpochs + " epochs");

    CostAndGradients costAndGradients = null;

    int iterationIndex = 0;
    
    int epochStartIndex = (lastEpochTrainingContext == null
        || lastEpochTrainingContext.getLastTrainingEpochIndex() == null) ? 0
            : (lastEpochTrainingContext.getLastTrainingEpochIndex() + 1);

    for (int epochIndex = epochStartIndex; 
        epochIndex < epochStartIndex + numberOfEpochs; epochIndex++) {

 
      if (trainingContext.getTrainingMiniBatchSize() == null) {
        costAndGradients =
            getCostAndGradients(trainingDataActivations, trainingLabelActivations, trainingContext);

        LOGGER.info("Epoch:" + epochIndex + " Cost:" + costAndGradients.getAverageCost());
        //Timings.printTimings();
        int batchIndex = epochIndex;
        
        List<AxonsGradient> averageTrainableAxonsGradients = costAndGradients.getAverageTrainableAxonsGradients();

        adjustConnectionWeights(trainingContext,
        		averageTrainableAxonsGradients,  epochIndex, batchIndex,
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
          int endColumnIndex =
              Math.min(startColumnIndex + miniBatchSize - 1, numberOfTrainingElements - 1);
          int[] columnsIndexes = new int[endColumnIndex - startColumnIndex + 1];
          for (int c = startColumnIndex; c <= endColumnIndex; c++) {
        	  columnsIndexes[c - startColumnIndex] = c;
          }
          try (InterrimMatrix dataBatch = activations.getColumns(columnsIndexes).asInterrimMatrix();
        		  InterrimMatrix labelBatch = trainingLabelActivations.getActivations(trainingContext.getMatrixFactory()).getColumns(columnsIndexes).asInterrimMatrix()) {
        
	          NeuronsActivation batchDataActivations =
	              new NeuronsActivationImpl(dataBatch, trainingDataActivations.getFeatureOrientation());
	
	          NeuronsActivation batchLabelActivations =
	              new NeuronsActivationImpl(labelBatch, trainingLabelActivations.getFeatureOrientation());
	
	          costAndGradients =
	              getCostAndGradients(batchDataActivations, batchLabelActivations, trainingContext);
	
	          LOGGER.info("Epoch:" + epochIndex + " batch " + batchIndex + " Cost:"
	              + costAndGradients.getAverageCost());
	          //Timings.printTimings();
	
	          List<AxonsGradient> averageTrainableAxonsGradients = costAndGradients.getAverageTrainableAxonsGradients();
	          
	          adjustConnectionWeights(trainingContext,
	        		  averageTrainableAxonsGradients , epochIndex, batchIndex,
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
        //Timings.printTimings();

        lastEpochTrainingContext = trainingContext;
      }
    }
  }
 
  protected CostAndGradientsImpl getCostAndGradients(NeuronsActivation inputActivations,
      NeuronsActivation desiredOutputActivations, C trainingContext) {
      
	  int numberOfTrainingExamples = inputActivations.getColumns();

	  
	  if (inputActivations
		        .getFeatureOrientation() != NeuronsActivationFeatureOrientation.ROWS_SPAN_FEATURE_SET) {
		      throw new IllegalArgumentException("Only neurons actiavation with ROWS_SPAN_FEATURE_SET "
		          + "orientation supported currently");
		    }
	  
	  if (desiredOutputActivations
		        .getFeatureOrientation() != NeuronsActivationFeatureOrientation.ROWS_SPAN_FEATURE_SET) {
		      throw new IllegalArgumentException("Only neurons actiavation with ROWS_SPAN_FEATURE_SET "
		          + "orientation supported currently");
		    }
	  
	  
    final CostFunction costFunction = getCostFunction();
    
    // Forward propagate the trainingDataActivations through the entire Network
    ForwardPropagation forwardPropagation =
        forwardPropagate(inputActivations, trainingContext);
    
 // Obtain the cost from the cost function
    LOGGER.debug("Calculating total cost function cost");
    float totalCost = costFunction.getTotalCost(
        desiredOutputActivations.getActivations(trainingContext.getMatrixFactory()),
        forwardPropagation.getOutput().getActivations(trainingContext.getMatrixFactory()));
   
    float totalRegularisationCost = forwardPropagation.getTotalRegularisationCost(trainingContext);
        
    float totalCostWithRegularisation = totalCost + totalRegularisationCost;
    
    CostFunctionGradient costFunctionGradient = 
        new DeltaRuleCostFunctionGradientImpl(costFunction, desiredOutputActivations, 
        		forwardPropagation.getOutput());
    
    // Back propagate the cost function gradient through the network
    BackPropagation backPropagation = forwardPropagation.backPropagate(costFunctionGradient, 
        trainingContext);
    
    
    // Obtain the gradients of each set of Axons we wish to train - for this example it is
    // all the Axons
    List<Supplier<AxonsGradient>> totalTrainableAxonsGradientSuppliers = backPropagation.getGradient().getTotalTrainableAxonsGradients();
    
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
    
    return new CostAndGradientsImpl(totalCostWithRegularisation, 
          totalTrainableAxonsGradients, numberOfTrainingExamples);
    
  }
  
  protected float getTrainingLearningRate(C trainingContext, int epochIndex, int batchIndex,
      int iterationIndex) {
    
    TrainingLearningRateAdjustmentStrategy adjustmentStrategy =
        trainingContext.getTrainingLearningRateAdjustmentStrategy();

    if (adjustmentStrategy != null) {
      return adjustmentStrategy.getTrainingLearningRate(trainingContext, 
           epochIndex, batchIndex, iterationIndex);
    }

    return trainingContext.getTrainingLearningRate();
  }

  protected AxonsGradient getAdjustedAxonsGradient(AxonsGradient axonsGradient, int axonsIndex, C trainingContext,
      int epochIndex, int batchIndex, int iterationIndex) {

    GradientDescentOptimisationStrategy optimisationStrategy =
        trainingContext.getGradientDescentOptimisationStrategy();

    if (optimisationStrategy != null) {
      return optimisationStrategy.getAdjustedAxonsGradient(axonsGradient, axonsIndex,
          trainingContext, epochIndex, batchIndex, iterationIndex);
    }
    return axonsGradient;
  }
  
  private void adjustConnectionWeights(C trainingContext,
      List<AxonsGradient> trainableAxonsGradients, int epochIndex, int batchIndex,
      int iterationIndex) {
    int axonsIndex = 0;
    for (AxonsGradient axonsGradient : trainableAxonsGradients) {
      TrainableAxons<?, ?, ?> trainableAxons = axonsGradient.getAxons();
      // Transpose the axon gradients into matrices that correspond to the orientation of the
      // connection weights ( COLUMNS_SPAN_FEATURE_SET )
      AxonsGradient adjustedAxonsGradient = getAdjustedAxonsGradient(axonsGradient, axonsIndex,
          trainingContext, epochIndex, batchIndex, iterationIndex);
      // Adjust the weights of each set of Axons by subtracting the learning-rate scaled
      // gradient matrices
      Matrix weightsGradient = adjustedAxonsGradient.getWeightsGradient();
      
      try (InterrimMatrix weightsAdjustment = weightsGradient.mul((float)
              getTrainingLearningRate(trainingContext, epochIndex, batchIndex, iterationIndex)).asInterrimMatrix()) {
    	  
    	  trainableAxons.adjustConnectionWeights(weightsAdjustment,  ConnectionWeightsAdjustmentDirection.SUBTRACTION);
      }
      
      if (trainableAxons.getLeftNeurons().hasBiasUnit()) {
    	  try (InterrimMatrix biasAdjustment = 
	    		  adjustedAxonsGradient.getLeftToRightBiasGradient().mul(
		                  getTrainingLearningRate(trainingContext, epochIndex, batchIndex, iterationIndex)).asInterrimMatrix()) {
    		  trainableAxons.adjustLeftToRightBiases(biasAdjustment, ConnectionWeightsAdjustmentDirection.SUBTRACTION);
    	  }
      }
      axonsIndex++;
    }
  }
 

  @Override
  public ForwardPropagation forwardPropagate(NeuronsActivation inputActivation,
      FeedForwardNeuralNetworkContext context) {
	
	//int endLayerIndex = context.getEndLayerIndex() == null ? (getNumberOfLayers() - 1) : context.getEndLayerIndex();  
	  
	LOGGER.debug("Forward propagating through FeedForwardNeuralNetwork");
	 	      
    // Forward propagate through the layers
	TrailingActivationFunctionDirectedComponentChainActivation activation = trailingActivationFunctionComponentChain.forwardPropagate(inputActivation, context.getDirectedComponentsContext());
    
    // Construct a forward propagation
    ForwardPropagation forwardPropagation 
        = new ForwardPropagationImpl(activation);

    if (context.getForwardPropagationListener() != null) {
      context.getForwardPropagationListener().onForwardPropagation(forwardPropagation);
    }
 
    return forwardPropagation;
  }
  
  /**
   * @return The default cost function for use by this Network.
   */
  protected CostFunction getCostFunction() {
	  
	  DifferentiableActivationFunction activationFunction = trailingActivationFunctionComponentChain.getFinalComponent().getActivationFunction();
	  
    if (activationFunction == null) {
      throw new UnsupportedOperationException(
          "Default cost function not yet defined for null activation function");
    }
    if (activationFunction instanceof SigmoidActivationFunction) {
      LOGGER.debug("Defaulting to use CrossEntropyCostFunction");
      return new CrossEntropyCostFunction();
    } else if (activationFunction instanceof SoftmaxActivationFunction) {
      LOGGER.debug("Defaulting to use MultiClassCrossEntropyCostFunction");
      return new MultiClassCrossEntropyCostFunction();
    } else if (activationFunction instanceof LinearActivationFunction) {
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
 
}
