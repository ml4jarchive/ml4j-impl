package org.ml4j.nn.axons.mocks;

import org.ml4j.nn.axons.AxonsContext;
import org.ml4j.nn.axons.FullyConnectedAxons;
import org.ml4j.nn.neurons.Neurons;
import org.ml4j.nn.neurons.NeuronsActivation;

public class AxonsMock implements FullyConnectedAxons {

  /**
   * Default serialization id.
   */
  private static final long serialVersionUID = 1L;

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
    throw new UnsupportedOperationException("Not yet implemented");
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
