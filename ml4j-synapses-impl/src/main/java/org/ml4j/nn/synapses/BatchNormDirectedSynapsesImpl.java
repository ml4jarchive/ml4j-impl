package org.ml4j.nn.synapses;

import org.ml4j.nn.activationfunctions.DifferentiableActivationFunction;
import org.ml4j.nn.axons.Axons;
import org.ml4j.nn.axons.ScaleAndShiftAxons;
import org.ml4j.nn.neurons.Neurons;
import org.ml4j.nn.neurons.NeuronsActivation;

/**
 * Default implementation of batch-norm DirectedSynapses
 * 
 * @author Michael Lavelle
 *
 * @param <L> The Neurons on the left hand side of these batch-norm DirectedSynapses.
 * @param <R> The Neurons on the right hand side of these batch-norm DirectedSynapses.
 */
public class BatchNormDirectedSynapsesImpl
      <L extends Neurons, R extends Neurons> implements DirectedSynapses<L, R> {

  /**
   * Default serialization id.
   */
  private static final long serialVersionUID = 1L;
  private L leftNeurons;
  private R rightNeurons;
  private ScaleAndShiftAxons scaleAndShiftAxons;
  
  /**
   * @param leftNeurons The left neurons.
   * @param rightNeurons The right neurons.
   * @param scaleAndShiftAxons The scale and shift axons.
   */
  public BatchNormDirectedSynapsesImpl(L leftNeurons, 
      R rightNeurons, ScaleAndShiftAxons scaleAndShiftAxons) {
    this.leftNeurons = leftNeurons;
    this.rightNeurons = rightNeurons;
    this.scaleAndShiftAxons = scaleAndShiftAxons;
 
  }
  
  @Override
  public DirectedSynapses<L, R> dup() {
    return new BatchNormDirectedSynapsesImpl<L, R>(leftNeurons, 
        rightNeurons, scaleAndShiftAxons.dup());
  }

  @Override
  public DirectedSynapsesGradient backPropagate(DirectedSynapsesActivation synapsesActivation,
      NeuronsActivation outerGradient, DirectedSynapsesContext context, boolean outer, double reg) {
    throw new UnsupportedOperationException("Not implemented yet");
  }

  @Override
  public DirectedSynapsesActivation forwardPropagate(DirectedSynapsesInput synapsesInput,
      DirectedSynapsesContext context) {
    throw new UnsupportedOperationException("Not implemented yet");    
  }
 
  @Override
  public L getLeftNeurons() {
    return leftNeurons;
  }

  @Override
  public R getRightNeurons() {
    return rightNeurons;
  }

  @Override
  public DifferentiableActivationFunction getActivationFunction() {
    return null;
  }

  @Override
  public Axons<?, ?, ?> getAxons() {
    return scaleAndShiftAxons;
  }
}
