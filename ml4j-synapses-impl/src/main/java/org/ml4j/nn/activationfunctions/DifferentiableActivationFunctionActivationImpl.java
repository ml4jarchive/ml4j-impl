package org.ml4j.nn.activationfunctions;

import org.ml4j.Matrix;
import org.ml4j.nn.costfunctions.CostFunctionGradient;
import org.ml4j.nn.neurons.NeuronsActivation;
import org.ml4j.nn.synapses.DirectedSynapsesContext;
import org.ml4j.nn.synapses.DirectedSynapsesGradient;

public class DifferentiableActivationFunctionActivationImpl
    implements DifferentiableActivationFunctionActivation {
  
  private NeuronsActivation input;
  private NeuronsActivation output;
  private DifferentiableActivationFunction activationFunction;

  /**
   * @param activationFunction The activation function that generated this activation.
   * @param input The input.
   * @param output The output.
   */
  public DifferentiableActivationFunctionActivationImpl(
      DifferentiableActivationFunction activationFunction, NeuronsActivation input,
      NeuronsActivation output) {
    this.input = input;
    this.output = output;
    this.activationFunction = activationFunction;
  }

  @Override
  public NeuronsActivation getInput() {
    return input;
  }

  @Override
  public NeuronsActivation getOutput() {
    return output;
  }

  @Override
  public ActivationFunctionGradient backPropagate(DirectedSynapsesGradient da,
      DirectedSynapsesContext context) {

    Matrix dz = null;

    Matrix activationGradient =
        activationFunction.activationGradient(this, context).getActivations().transpose();

    dz = da.getOutput().getActivations().mul(activationGradient);

    return new ActivationFunctionGradientImpl((new NeuronsActivation(dz, 
        da.getOutput().getFeatureOrientation())));
  }

  @Override
  public ActivationFunctionGradient backPropagate(CostFunctionGradient da, 
      DirectedSynapsesContext context) {
    return da.backPropagateThroughFinalActivationFunction(activationFunction);
  }

  @Override
  public DifferentiableActivationFunction getActivationFunction() {
    return activationFunction;
  }

}
