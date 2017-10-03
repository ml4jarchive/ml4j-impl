package org.ml4j.nn.axons.mocks;

import org.ml4j.nn.axons.AxonsContext;
import org.ml4j.nn.axons.FullyConnectedAxons;
import org.ml4j.nn.neurons.Neurons;
import org.ml4j.nn.neurons.NeuronsActivation;
import org.ml4j.nn.neurons.NeuronsActivationFeatureOrientation;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class AxonsMock implements FullyConnectedAxons {

  /**
   * Default serialization id.
   */
  private static final long serialVersionUID = 1L;

  private static final Logger LOGGER = LoggerFactory.getLogger(
      AxonsMock.class);
  
  private Neurons leftNeurons;
  private Neurons rightNeurons;
  
  public AxonsMock(Neurons leftNeurons, Neurons rightNeurons) {
    this.leftNeurons = leftNeurons;
    this.rightNeurons = rightNeurons;
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
    LOGGER.debug("Mock pushing left to right through Axons");
    return new NeuronsActivation(axonsContext.getMatrixFactory().createZeros(
        leftNeuronsActivation.getActivations().getRows(), 
        rightNeurons.getNeuronCountIncludingBias()), 
        rightNeurons.hasBiasUnit(), NeuronsActivationFeatureOrientation.COLUMNS_SPAN_FEATURE_SET);
  }

  @Override
  public NeuronsActivation pushRightToLeft(NeuronsActivation rightNeuronsActivation,
      AxonsContext axonsContext) {
    throw new UnsupportedOperationException("Not yet implemented");
  }

  @Override
  public FullyConnectedAxons dup() {
    return new AxonsMock(leftNeurons, rightNeurons);
  }
}
