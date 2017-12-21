package org.ml4j.nn.synapses;

import org.ml4j.Matrix;
import org.ml4j.nn.activationfunctions.DifferentiableActivationFunction;
import org.ml4j.nn.axons.Axons;
import org.ml4j.nn.axons.ScaleAndShiftAxons;
import org.ml4j.nn.graph.DirectedDipoleGraph;
import org.ml4j.nn.graph.DirectedDipoleGraphImpl;
import org.ml4j.nn.neurons.Neurons;

/**
 * Default implementation of batch-norm DirectedSynapses
 * 
 * @author Michael Lavelle
 *
 * @param <L> The Neurons on the left hand side of these batch-norm DirectedSynapses.
 * @param <R> The Neurons on the right hand side of these batch-norm DirectedSynapses.
 */
public class BatchNormDirectedSynapsesImpl
      <L extends Neurons, R extends Neurons> implements BatchNormDirectedSynapses<L, R> {

  /**
   * Default serialization id.
   */
  private static final long serialVersionUID = 1L;
  private L leftNeurons;
  private R rightNeurons;
  private ScaleAndShiftAxons scaleAndShiftAxons;
  private DifferentiableActivationFunction activationFunction;
  
  /**
   * @param leftNeurons The left neurons.
   * @param rightNeurons The right neurons.
   * @param scaleAndShiftAxons The scale and shift axons.
   * @param activationFunction The activation function to apply after the batch norm processing.
   */
  public BatchNormDirectedSynapsesImpl(L leftNeurons, 
      R rightNeurons, ScaleAndShiftAxons scaleAndShiftAxons, 
      DifferentiableActivationFunction activationFunction) {
    this.leftNeurons = leftNeurons;
    this.rightNeurons = rightNeurons;
    this.scaleAndShiftAxons = scaleAndShiftAxons;
    this.activationFunction = activationFunction;
 
  }
  
  @Override
  public DirectedSynapses<L, R> dup() {
    return new BatchNormDirectedSynapsesImpl<L, R>(leftNeurons, 
        rightNeurons, scaleAndShiftAxons.dup(), activationFunction);
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
    return activationFunction;
  }

  @Override
  public double getBetaForExponentiallyWeightedAverages() {
    throw new UnsupportedOperationException("Not implemented yet");    
  }

  @Override
  public Matrix getExponentiallyWeightedAverageInputFeatureMeans() {
    throw new UnsupportedOperationException("Not implemented yet");    
  }

  @Override
  public Matrix getExponentiallyWeightedAverageInputFeatureVariances() {
    throw new UnsupportedOperationException("Not implemented yet");    
  }

  @Override
  public void setExponentiallyWeightedAverageInputFeatureMeans(Matrix arg0) {
    throw new UnsupportedOperationException("Not implemented yet");    
  }

  @Override
  public void setExponentiallyWeightedAverageInputFeatureVariances(Matrix arg0) {
    throw new UnsupportedOperationException("Not implemented yet");    
  }

  @Override
  public DirectedDipoleGraph<Axons<?, ?, ?>> getAxonsGraph() {
    return new DirectedDipoleGraphImpl<Axons<?, ?, ?>>(scaleAndShiftAxons);
  }

  @Override
  public Axons<?, ?, ?> getPrimaryAxons() {
    return scaleAndShiftAxons;
  }
}
