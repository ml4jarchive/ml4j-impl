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

package org.ml4j.nn.synapses;

import org.ml4j.nn.activationfunctions.ActivationFunction;
import org.ml4j.nn.axons.Axons;
import org.ml4j.nn.axons.AxonsActivation;
import org.ml4j.nn.neurons.Neurons;
import org.ml4j.nn.neurons.NeuronsActivation;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Default implementation of UndirectedSynapses.
 * 
 * @author Michael Lavelle
 */
public class UndirectedSynapsesImpl<L extends Neurons, R extends Neurons>
    implements UndirectedSynapses<L, R> {

  /**
   * Default serialization id.
   */
  private static final long serialVersionUID = 1L;

  private static final Logger LOGGER = LoggerFactory.getLogger(UndirectedSynapsesImpl.class);

  private Axons<? extends L, ? extends R, ?> axons;
  private ActivationFunction leftActivationFunction;
  private ActivationFunction rightActivationFunction;


  /**
   * Create a new implementation of DirectedSynapses.
   * 
   * @param axons The Axons within these synapses
   * @param leftActivationFunction The activation function on the left hand side of these synapses
   * @param rightActivationFunction The activation function on the right hand side of these synapses
   */
  public UndirectedSynapsesImpl(Axons<? extends L, ? extends R, ?> axons,
      ActivationFunction leftActivationFunction, ActivationFunction rightActivationFunction) {
    super();
    this.axons = axons;
    this.leftActivationFunction = leftActivationFunction;
    this.rightActivationFunction = rightActivationFunction;
  }

  @Override
  public Axons<? extends L, ? extends R, ?> getAxons() {
    return axons;
  }

  @Override
  public UndirectedSynapses<L, R> dup() {
    return new UndirectedSynapsesImpl<L, R>(axons.dup(), leftActivationFunction,
        rightActivationFunction);
  }


  @Override
  public ActivationFunction getLeftActivationFunction() {
    return leftActivationFunction;
  }

  @Override
  public ActivationFunction getRightActivationFunction() {
    return leftActivationFunction;
  }


  @Override
  public UndirectedSynapsesActivation pushLeftToRight(UndirectedSynapsesInput input,
      UndirectedSynapsesContext synapsesContext) {

    NeuronsActivation inputNeuronsActivation = input.getInput();

    LOGGER.debug("Pushing left to right through UndirectedSynapses");
    AxonsActivation axonsActivation =
        axons.pushLeftToRight(inputNeuronsActivation, null, synapsesContext.createAxonsContext());

    NeuronsActivation axonsOutputActivation = axonsActivation.getOutput();

    NeuronsActivation outputNeuronsActivation =
        rightActivationFunction.activate(axonsOutputActivation, synapsesContext);

    return new UndirectedSynapsesActivationImpl(this, inputNeuronsActivation, axonsActivation,
        outputNeuronsActivation);
  }

  @Override
  public UndirectedSynapsesActivation pushRightToLeft(UndirectedSynapsesInput input,
      UndirectedSynapsesContext synapsesContext) {

    NeuronsActivation inputNeuronsActivation = input.getInput();

    LOGGER.debug("Pushing right to left through UndirectedSynapses");
    AxonsActivation axonsActivation =
        axons.pushLeftToRight(inputNeuronsActivation, null, synapsesContext.createAxonsContext());

    NeuronsActivation axonsOutputActivation = axonsActivation.getOutput();

    NeuronsActivation outputNeuronsActivation =
        leftActivationFunction.activate(axonsOutputActivation, synapsesContext);

    return new UndirectedSynapsesActivationImpl(this, inputNeuronsActivation, axonsActivation,
        outputNeuronsActivation);
  }

  @Override
  public L getLeftNeurons() {
    return axons.getLeftNeurons();
  }

  @Override
  public R getRightNeurons() {
    return axons.getRightNeurons();
  }
}
