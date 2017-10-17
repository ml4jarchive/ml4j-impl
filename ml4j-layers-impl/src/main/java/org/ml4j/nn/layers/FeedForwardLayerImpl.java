/*
 * Copyright 2017 the original author or authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
 * in compliance with the License. You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software distributed under the License
 * is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
 * or implied. See the License for the specific language governing permissions and limitations under
 * the License.
 */

package org.ml4j.nn.layers;

import org.ml4j.nn.activationfunctions.DifferentiableActivationFunction;
import org.ml4j.nn.axons.Axons;
import org.ml4j.nn.neurons.Neurons;
import org.ml4j.nn.neurons.NeuronsActivation;
import org.ml4j.nn.synapses.DirectedSynapses;

import java.util.List;

/**
 * The default ml4j FeedForwardLayer implementation.
 * 
 * @author Michael Lavelle
 */
public class FeedForwardLayerImpl
    implements FeedForwardLayer<Axons<?, ?, ?>, FeedForwardLayerImpl> {

  /**
   * Default serialization id.
   */
  private static final long serialVersionUID = 1L;

  public FeedForwardLayerImpl(Neurons inputNeurons, Neurons outputNeurons,
      DifferentiableActivationFunction primaryActivationFunction) {
    throw new UnsupportedOperationException("Not implemented yet");
  }

  @Override
  public FeedForwardLayerImpl dup() {
    throw new UnsupportedOperationException("Not implemented yet");
  }

  @Override
  public int getInputNeuronCount() {
    throw new UnsupportedOperationException("Not implemented yet");
  }

  @Override
  public int getOutputNeuronCount() {
    throw new UnsupportedOperationException("Not implemented yet");
  }

  @Override
  public Axons<?, ?, ?> getPrimaryAxons() {
    throw new UnsupportedOperationException("Not implemented yet");
  }

  @Override
  public NeuronsActivation getOptimalInputForOutputNeuron(int outputNeuronIndex,
      DirectedLayerContext directedLayerContext) {
    throw new UnsupportedOperationException("Not implemented yet");
  }


  @Override
  public DifferentiableActivationFunction getPrimaryActivationFunction() {
    throw new UnsupportedOperationException("Not implemented yet");
  }

  @Override
  public DirectedLayerActivation forwardPropagate(NeuronsActivation inputNeuronsActivation,
      DirectedLayerContext directedLayerContext) {
    throw new UnsupportedOperationException("Not implemented yet");
  }

  @Override
  public List<DirectedSynapses<?>> getSynapses() {
    throw new UnsupportedOperationException("Not implemented yet");
  }
}
