package org.ml4j.nn.axons;

import org.ml4j.Matrix;
import org.ml4j.nn.axons.AxonsContext;
import org.ml4j.nn.axons.FullyConnectedAxons;
import org.ml4j.nn.neurons.Neurons;
import org.ml4j.nn.neurons.NeuronsActivation;
import org.ml4j.nn.neurons.NeuronsActivationFeatureOrientation;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class AxonsImpl implements FullyConnectedAxons {

  /**
   * Default serialization id.
   */
  private static final long serialVersionUID = 1L;

  private static final Logger LOGGER = LoggerFactory.getLogger(
      AxonsImpl.class);
  
  private Neurons leftNeurons;
  private Neurons rightNeurons;
  private Matrix connectionWeights;
  
  /**
   * Construct a new mock Axons instance.
   * 
   * @param leftNeurons The Neurons on the left hand side of these Axons
   * @param rightNeurons The Neurons on the right hand side of these Axons
   * @param connectionWeights The connection weights Matrix
   */
  public AxonsImpl(Neurons leftNeurons, Neurons rightNeurons, Matrix connectionWeights) {
    this.leftNeurons = leftNeurons;
    this.rightNeurons = rightNeurons;
    this.connectionWeights = connectionWeights;
  }
  
  @Override
  public Neurons getLeftNeurons() {
    return leftNeurons;
  }

  @Override
  public Neurons getRightNeurons() {
    return rightNeurons;
  }

  @Override
  public NeuronsActivation pushLeftToRight(NeuronsActivation leftNeuronsActivation,
      AxonsContext axonsContext) {
    LOGGER.debug("Pushing left to right through Axons");
    if (leftNeuronsActivation.getFeatureOrientation()
            != NeuronsActivationFeatureOrientation.COLUMNS_SPAN_FEATURE_SET) {
      throw new IllegalArgumentException("Only neurons actiavation with COLUMNS_SPAN_FEATURE_SET "
          + "orientation supported currently");
    }
    Matrix outputMatrix =
        leftNeuronsActivation.withBiasUnit(leftNeurons.hasBiasUnit(), axonsContext).getActivations()
            .mmul(connectionWeights);
    return new NeuronsActivation(outputMatrix, rightNeurons.hasBiasUnit(),
        leftNeuronsActivation.getFeatureOrientation()).withBiasUnit(rightNeurons.hasBiasUnit(),
            axonsContext);
  }

  @Override
  public NeuronsActivation pushRightToLeft(NeuronsActivation rightNeuronsActivation,
      AxonsContext axonsContext) {
    throw new UnsupportedOperationException("Not yet implemented");
  }

  @Override
  public FullyConnectedAxons dup() {
    return new AxonsImpl(leftNeurons, rightNeurons, connectionWeights.dup());
  }

  public void setConnectionWeights(Matrix connectionWeights) {
    this.connectionWeights = connectionWeights;
  }

  public Matrix getConnectionWeights() {
    return connectionWeights;
  }
  
}
