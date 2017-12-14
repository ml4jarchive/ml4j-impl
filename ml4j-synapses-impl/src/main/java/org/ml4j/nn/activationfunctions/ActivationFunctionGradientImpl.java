package org.ml4j.nn.activationfunctions;

import org.ml4j.nn.neurons.NeuronsActivation;

public class ActivationFunctionGradientImpl implements ActivationFunctionGradient {

  private NeuronsActivation output;
  
  public ActivationFunctionGradientImpl(NeuronsActivation gradient) {
    this.output = gradient;
  }
  

  @Override
  public NeuronsActivation getOutput() {
    return output;
  }

}
