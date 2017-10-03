package org.ml4j.nn.layers.mocks;

import org.ml4j.nn.layers.DirectedLayerActivation;
import org.ml4j.nn.neurons.NeuronsActivation;

public class DirectedLayerActivationMock implements DirectedLayerActivation {

  private NeuronsActivation outputActivation;
  
  public DirectedLayerActivationMock(NeuronsActivation outputActivation) {
    this.outputActivation = outputActivation;
  }
  
  @Override
  public NeuronsActivation getOutput() {
    return outputActivation;
  }

}
