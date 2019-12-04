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

package org.ml4j.nn.supervised;

import java.util.Arrays;
import java.util.List;

import org.ml4j.EditableMatrix;
import org.ml4j.Matrix;
import org.ml4j.MatrixFactory;
import org.ml4j.nn.CostAndGradientsImpl;
import org.ml4j.nn.FeedForwardNeuralNetworkBase;
import org.ml4j.nn.FeedForwardNeuralNetworkContext;
import org.ml4j.nn.ForwardPropagation;
import org.ml4j.nn.components.ChainableDirectedComponent;
import org.ml4j.nn.components.ChainableDirectedComponentActivation;
import org.ml4j.nn.components.DirectedComponentChain;
import org.ml4j.nn.components.DirectedComponentsContext;
import org.ml4j.nn.components.defaults.DefaultDirectedComponentChainImpl;
import org.ml4j.nn.neurons.NeuronsActivation;
import org.ml4j.nn.neurons.NeuronsActivationFeatureOrientation;

/**
 * Default implementation of SupervisedFeedForwardNeuralNetwork.
 *
 * @author Michael Lavelle
 */
public class SupervisedFeedForwardNeuralNetworkImpl 
    extends FeedForwardNeuralNetworkBase<FeedForwardNeuralNetworkContext, DirectedComponentChain<NeuronsActivation, ?, ?, ?>,
    SupervisedFeedForwardNeuralNetwork> implements SupervisedFeedForwardNeuralNetwork {
  
  /**
   * Default serialization id.
   */
  private static final long serialVersionUID = 1L;
  
  /**
   * Constructor for a multi-layer supervised FeedForwardNeuralNetwork.
   * 
   * @param layers The layers
   */
  public SupervisedFeedForwardNeuralNetworkImpl(DirectedComponentChain<NeuronsActivation, ?, ?, ?> initialisingComponentChain) {
    super(initialisingComponentChain);
  }
  
  /**
   * Constructor for a multi-layer supervised FeedForwardNeuralNetwork.
   * 
   * @param layers The layers
   */
  public SupervisedFeedForwardNeuralNetworkImpl(List<ChainableDirectedComponent<NeuronsActivation, ? extends ChainableDirectedComponentActivation<NeuronsActivation>, ?>> componentList) {
    super(new DefaultDirectedComponentChainImpl<>(componentList));
  }
  
  /**
   * Constructor for a multi-layer supervised FeedForwardNeuralNetwork.
   * 
   * @param layers The layers
   */
  public SupervisedFeedForwardNeuralNetworkImpl(ChainableDirectedComponent<NeuronsActivation, ? extends ChainableDirectedComponentActivation<NeuronsActivation>, ?>... componentList) {
    super(new DefaultDirectedComponentChainImpl<>(Arrays.asList(componentList)));
  }

  @Override
  public void train(NeuronsActivation trainingDataActivations,
      NeuronsActivation trainingLabelActivations, FeedForwardNeuralNetworkContext trainingContext) {
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
    super.train(trainingDataActivations, trainingLabelActivations, trainingContext);
  }
  
  /**
   * Return the prediction accuracy.
   * 
   * @param inputActivations The input activations.
   * @param desiredClassificationActivations The desired prediction activations.
   * @param context The context.
   * @return The accuracy
   */
  @Override
  public float getClassificationAccuracy(NeuronsActivation inputActivations,
      NeuronsActivation desiredClassificationActivations, FeedForwardNeuralNetworkContext context) {

    // Forward propagate the trainingDataActivations
    ForwardPropagation forwardPropagation = forwardPropagate(inputActivations, context);

    Matrix predictions = getClassifications(forwardPropagation.getOutput().getActivations(context.getMatrixFactory()).transpose(),
        context.getMatrixFactory());

    return computeAccuracy(predictions, desiredClassificationActivations.getActivations(context.getMatrixFactory()).transpose());
  }

  private Matrix getClassifications(Matrix outputActivations, MatrixFactory matrixFactory) {

    EditableMatrix predictions = 
        matrixFactory.createZeros(outputActivations.getRows(), outputActivations.getColumns()).asEditableMatrix();
    for (int row = 0; row < outputActivations.getRows(); row++) {

      int index = outputActivations.getRow(row).argmax();
      predictions.put(row, index, 1);
    }
    return predictions;
  }

  /**
   * Helper function to compute the accuracy of predictions using calculated predictions predictions
   * and correct output matrix.
   *
   * @param predictions The predictions
   * @param Y The desired output labels
   * @return The accuracy of the network
   */
  protected float computeAccuracy(Matrix predictions, Matrix outputs) {
    return ((predictions.mul(outputs)).sum()) * 100 / outputs.getRows();
  }

  @Override
  public SupervisedFeedForwardNeuralNetwork dup() {
	  throw new UnsupportedOperationException("Not yet implemented");
    //return new SupervisedFeedForwardNeuralNetworkImpl2(getLayer(0), getLayer(1));
  }
  
  @Override
  public CostAndGradientsImpl getCostAndGradients(NeuronsActivation inputActivations,
      NeuronsActivation desiredOutpuActivations, FeedForwardNeuralNetworkContext trainingContext) {
    return super.getCostAndGradients(inputActivations, desiredOutpuActivations, trainingContext);
  }

@Override
public List<ChainableDirectedComponent<NeuronsActivation, ? extends ChainableDirectedComponentActivation<NeuronsActivation>, ?>> decompose() {
	throw new UnsupportedOperationException(); 

}

@Override
public FeedForwardNeuralNetworkContext getContext(DirectedComponentsContext arg0, int arg1) {
	throw new UnsupportedOperationException(); 
}
}
