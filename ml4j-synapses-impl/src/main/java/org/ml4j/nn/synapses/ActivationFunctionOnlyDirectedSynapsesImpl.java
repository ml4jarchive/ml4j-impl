package org.ml4j.nn.synapses;

import org.ml4j.Matrix;
import org.ml4j.nn.activationfunctions.DifferentiableActivationFunction;
import org.ml4j.nn.axons.Axons;
import org.ml4j.nn.neurons.Neurons;
import org.ml4j.nn.neurons.NeuronsActivation;

public class ActivationFunctionOnlyDirectedSynapsesImpl
      <L extends Neurons, R extends Neurons> implements DirectedSynapses<L, R> {

  /**
   * Default serialization id.
   */
  private static final long serialVersionUID = 1L;
  private L leftNeurons;
  private R rightNeurons;
  private DifferentiableActivationFunction activationFunction;


  /**
   * @param leftNeurons The left neurons
   * @param rightNeurons The right neurons
   * @param activationFunction The activation function.
   */
  public ActivationFunctionOnlyDirectedSynapsesImpl(L leftNeurons, R rightNeurons, 
          DifferentiableActivationFunction activationFunction) {
    this.leftNeurons = leftNeurons;
    this.rightNeurons = rightNeurons;
    this.activationFunction = activationFunction;
  }
  
  @Override
  public DirectedSynapses<L, R> dup() {
    return new ActivationFunctionOnlyDirectedSynapsesImpl<L, R>(leftNeurons, 
        rightNeurons, activationFunction);
  }

  @Override
  public DirectedSynapsesGradient backPropagate(DirectedSynapsesActivation activation,
      NeuronsActivation da, DirectedSynapsesContext context, 
      boolean outerMostSynapses, double arg4) {
    
    //LOGGER.debug("Back propagating through synapses activation....");
    
    if (da.isBiasUnitIncluded()) {
      throw new IllegalArgumentException("Back propagated deltas must not contain bias unit");
    }
    
    if (rightNeurons.hasBiasUnit()) {
      throw new IllegalStateException(
          "Backpropagation through axons with a rhs bias unit not supported");
    }
    
    if (activation.getAxonsActivation() != null) {
      throw new IllegalStateException(
          "The synapses activation is not expected to contain an AxonsActivation");
    }
    
    NeuronsActivation activationFunctionInput = activation.getInput();
    
    Matrix dz = null;
    
    if (outerMostSynapses) {
      dz = da.getActivations();
    } else {
      Matrix activationGradient = activationFunction
          .activationGradient(activationFunctionInput.withBiasUnit(false, context), context)
          .getActivations();

      dz = da.getActivations().mul(activationGradient);
    }
  
    if (da.getFeatureCountIncludingBias() != rightNeurons
        .getNeuronCountExcludingBias()) {
      throw new IllegalArgumentException("Expected feature count to be:"
          + rightNeurons.getNeuronCountExcludingBias() + " but was:"
          + da.getFeatureCountIncludingBias());
    }
    
    // Does not contain output bias unit
    NeuronsActivation dzN = new NeuronsActivation(dz, 
        false,
        da.getFeatureOrientation());

 
    return new DirectedSynapsesGradientImpl(dzN, 
        null);
    
    
  }

  @Override
  public DirectedSynapsesActivation forwardPropagate(DirectedSynapsesInput input,
      DirectedSynapsesContext synapsesContext) {
  
    NeuronsActivation inputNeuronsActivation = input.getInput();
    
    if (!inputNeuronsActivation.isBiasUnitIncluded() && leftNeurons.hasBiasUnit()) {
      inputNeuronsActivation = inputNeuronsActivation.withBiasUnit(false, synapsesContext);
    }
   
    //LOGGER.debug("Forward propagating through DirectedSynapses");

    
    NeuronsActivation activationInput = inputNeuronsActivation;
    
    NeuronsActivation outputNeuronsActivation = 
        activationFunction.activate(activationInput, 
            synapsesContext).withBiasUnit(rightNeurons.hasBiasUnit(), synapsesContext);
    
    return new DirectedSynapsesActivationImpl(this, 
        inputNeuronsActivation, null, outputNeuronsActivation);
    
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
  public Axons<?, ?, ?> getAxons() {
    return null;
  }
}
