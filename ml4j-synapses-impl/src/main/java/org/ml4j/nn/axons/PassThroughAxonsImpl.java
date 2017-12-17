package org.ml4j.nn.axons;

import org.ml4j.Matrix;
import org.ml4j.nn.neurons.Neurons;
import org.ml4j.nn.neurons.NeuronsActivation;

/**
 * Pass through Axons.
 * 
 * @author Michael Lavelle
 */
public class PassThroughAxonsImpl implements Axons<Neurons, Neurons, PassThroughAxonsImpl> {

  /**
   * Default serialization id.
   */
  private static final long serialVersionUID = 1L;

  private Neurons leftNeurons;
  private Neurons rightNeurons;

  /**
   * @param leftNeurons The left neurons.
   * @param rightNeurons The right neurons.
   */
  public PassThroughAxonsImpl(Neurons leftNeurons, Neurons rightNeurons) {
    this.leftNeurons = leftNeurons;
    this.rightNeurons = rightNeurons;
    if (leftNeurons.getNeuronCountIncludingBias() != rightNeurons.getNeuronCountIncludingBias()) {
      throw new IllegalArgumentException("Left neuron and right neurons counts must be the same");
    }
    if (leftNeurons.hasBiasUnit() != rightNeurons.hasBiasUnit()) {
      throw new IllegalArgumentException(
          "Left neuron and right neurons bias unit presence must be the same");
    }
  }

  @Override
  public PassThroughAxonsImpl dup() {
    return new PassThroughAxonsImpl(leftNeurons, rightNeurons);
  }

  @Override
  public Matrix getDetachedConnectionWeights() {
    throw new UnsupportedOperationException("No connection weights in pass through axons");
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
  public boolean isTrainable(AxonsContext context) {
    return false;
  }

  @Override
  public AxonsActivation pushLeftToRight(NeuronsActivation input, AxonsActivation arg1,
      AxonsContext arg2) {
    return new AxonsActivationImpl(this, null, null, input);
  }

  @Override
  public AxonsActivation pushRightToLeft(NeuronsActivation input, AxonsActivation arg1,
      AxonsContext arg2) {
    return new AxonsActivationImpl(this, null, null, input);
  }
}
