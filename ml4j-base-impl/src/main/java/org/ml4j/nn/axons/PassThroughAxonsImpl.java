package org.ml4j.nn.axons;

import org.ml4j.nn.neurons.Neurons;
import org.ml4j.nn.neurons.NeuronsActivation;

/**
 * Pass through Axons.
 * 
 * @author Michael Lavelle
 */
public class PassThroughAxonsImpl<N extends Neurons> implements Axons<N, N, PassThroughAxonsImpl<N>> {

  /**
   * Default serialization id.
   */
  private static final long serialVersionUID = 1L;

  private N leftNeurons;
  private N rightNeurons;

  /**
   * @param leftNeurons The left neurons.
   * @param rightNeurons The right neurons.
   */
  public PassThroughAxonsImpl(N leftNeurons, N rightNeurons) {
    this.leftNeurons = leftNeurons;
    this.rightNeurons = rightNeurons;
    
    if (leftNeurons.getNeuronCountExcludingBias() != rightNeurons.getNeuronCountExcludingBias()) {
      throw new IllegalArgumentException("Left neuron and right neurons counts must be the same"
          + leftNeurons.getNeuronCountIncludingBias() + ":"
          + rightNeurons.getNeuronCountIncludingBias());
    }
    if (leftNeurons.hasBiasUnit() != rightNeurons.hasBiasUnit()) {
      throw new IllegalArgumentException(
          "Left neuron and right neurons bias unit presence must be the same");
    }
    if (leftNeurons.hasBiasUnit()) {
      throw new IllegalArgumentException("Left neurons with bias unit not supported");
    }
  }

  @Override
  public PassThroughAxonsImpl<N> dup() {
    return new PassThroughAxonsImpl<N>(leftNeurons, rightNeurons);
  }

  @Override
  public N getLeftNeurons() {
    return leftNeurons;
  }

  @Override
  public N getRightNeurons() {
    return rightNeurons;
  }

  @Override
  public boolean isTrainable(AxonsContext context) {
    return false;
  }

  @Override
  public AxonsActivation pushLeftToRight(NeuronsActivation input, AxonsActivation arg1,
      AxonsContext arg2) {
    return new AxonsActivationImpl(this, null, null, input, leftNeurons, rightNeurons);
  }

  @Override
  public AxonsActivation pushRightToLeft(NeuronsActivation input, AxonsActivation arg1,
      AxonsContext arg2) {
    return new AxonsActivationImpl(this, null, null, input, leftNeurons, rightNeurons);
  }
}
