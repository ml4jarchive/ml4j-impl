/*
 * Copyright 2017 the original author or authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.ml4j.nn.layers;

import org.ml4j.Matrix;
import org.ml4j.MatrixFactory;
import org.ml4j.nn.activationfunctions.DifferentiableActivationFunction;
import org.ml4j.nn.axons.Axons;
import org.ml4j.nn.axons.AxonsImpl;
import org.ml4j.nn.layers.DirectedLayerActivation;
import org.ml4j.nn.layers.DirectedLayerContext;
import org.ml4j.nn.layers.FeedForwardLayer;
import org.ml4j.nn.neurons.Neurons;
import org.ml4j.nn.neurons.NeuronsActivation;
import org.ml4j.nn.neurons.NeuronsActivationFeatureOrientation;
import org.ml4j.nn.synapses.DirectedSynapses;
import org.ml4j.nn.synapses.DirectedSynapsesActivation;
import org.ml4j.nn.synapses.DirectedSynapsesImpl;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.List;

/**
 * A default implementation of FeedForwardLayer.
 * 
 * @author Michael Lavelle
 */
public class FeedForwardLayerImpl implements FeedForwardLayer<Axons<?, ?, ?>, 
    FeedForwardLayerImpl> {

  /**
   * Default serialization id.
   */
  private static final long serialVersionUID = 1L;
 
  private static final Logger LOGGER = 
      LoggerFactory.getLogger(FeedForwardLayerImpl.class);

  private Axons<?, ?, ?> primaryAxons;
  
  private DifferentiableActivationFunction primaryActivationFunction;
  
  /**
   * @param inputNeurons The input Neurons.
   * @param outputNeurons The output Neurons
   * @param primaryActivationFunction The primary activation function.
   */
  public FeedForwardLayerImpl(Neurons inputNeurons, Neurons outputNeurons,
      DifferentiableActivationFunction primaryActivationFunction, MatrixFactory matrixFactory) {
    this(
        new AxonsImpl(inputNeurons, outputNeurons,
            createInitialAxonConnectionWeights(inputNeurons, outputNeurons, matrixFactory)),
        primaryActivationFunction);
  }
  
  /**
   * Obtain the initial axon connection weights.
   * 
   * @param inputNeurons The input Neurons
   * @param outputNeurons The output Neurons
   * @param matrixFactory The matrix factory
   * @return The initial conn
   */
  private static Matrix createInitialAxonConnectionWeights(Neurons inputNeurons,
      Neurons outputNeurons, MatrixFactory matrixFactory) {
   
    LOGGER.debug("Initialising Axon weights...");
    
    Matrix weights = matrixFactory.createRandn(inputNeurons.getNeuronCountIncludingBias(),
        outputNeurons.getNeuronCountIncludingBias());

    double scalingFactor = 
        Math.sqrt(2 / ((double)inputNeurons.getNeuronCountIncludingBias()));
    
    return weights.mul(scalingFactor);
  }

  protected FeedForwardLayerImpl(Axons<?, ?, ?> primaryAxons, 
      DifferentiableActivationFunction activationFunction) {
    this.primaryAxons = primaryAxons;
    this.primaryActivationFunction = activationFunction;
  }

  @Override
  public FeedForwardLayerImpl dup() {
    return new FeedForwardLayerImpl(primaryAxons.dup(), primaryActivationFunction);
  }

  @Override
  public int getInputNeuronCount() {
    return primaryAxons.getLeftNeurons().getNeuronCountIncludingBias();
  }

  @Override
  public int getOutputNeuronCount() {
    return primaryAxons.getRightNeurons().getNeuronCountIncludingBias();
  }

  @Override
  public Axons<?, ?, ?> getPrimaryAxons() {
    return primaryAxons;
  }

  @Override
  public NeuronsActivation getOptimalInputForOutputNeuron(int outputNeuronIndex,
      DirectedLayerContext directedLayerContext) {
    LOGGER.debug("Mock obtaining optimal input for output neuron with index:" + outputNeuronIndex);
    int countJ = getPrimaryAxons().getLeftNeurons().getNeuronCountExcludingBias();
    double[] maximisingInputFeatures = new double[countJ];
    Matrix weights = ((AxonsImpl) getPrimaryAxons()).getConnectionWeights();
    boolean hasBiasUnit = getPrimaryAxons().getLeftNeurons().hasBiasUnit();

    for (int j = 0; j < countJ; j++) {
      double wij = getWij(j, outputNeuronIndex, weights, hasBiasUnit);
      double sum = 0;

      if (wij != 0) {

        for (int j2 = 0; j2 < countJ; j2++) {
          double weight = getWij(j2, outputNeuronIndex, weights, hasBiasUnit);
          if (weight != 0) {
            sum = sum + Math.pow(weight, 2);
          }
        }
        sum = Math.sqrt(sum);
      }
      maximisingInputFeatures[j] = wij / sum;
    }
    return new NeuronsActivation(
        directedLayerContext.getMatrixFactory()
            .createMatrix(new double[][] {maximisingInputFeatures}),
        false, NeuronsActivationFeatureOrientation.COLUMNS_SPAN_FEATURE_SET);
  }
  
  private double getWij(int indI, int indJ, Matrix weights, boolean hasBiasUnit) {
    int indICorrected = indI + (hasBiasUnit ? 1 : 0);
    return weights.get(indICorrected, indJ);
  }

  @Override
  public DifferentiableActivationFunction getPrimaryActivationFunction() {
    return primaryActivationFunction;
  }

  @Override
  public DirectedLayerActivation forwardPropagate(NeuronsActivation inputNeuronsActivation,
      DirectedLayerContext directedLayerContext) {
    LOGGER.debug("Forward propagating through layer");
    
    NeuronsActivation inFlightNeuronsActivation = inputNeuronsActivation;
    List<DirectedSynapsesActivation> synapseActivations = new ArrayList<>();
    int synapsesIndex = 0;
    for (DirectedSynapses<?> synapses : getSynapses()) {
      DirectedSynapsesActivation inFlightNeuronsSynapseActivation = 
          synapses.forwardPropagate(inFlightNeuronsActivation, 
              directedLayerContext.createSynapsesContext(synapsesIndex++));
      synapseActivations.add(inFlightNeuronsSynapseActivation);
      inFlightNeuronsActivation = inFlightNeuronsSynapseActivation.getOutput();
    }
 
    return new DirectedLayerActivationImpl(this, synapseActivations, 
        inFlightNeuronsActivation);
  }

  @Override
  public List<DirectedSynapses<?>> getSynapses() {
    List<DirectedSynapses<?>> synapses = new ArrayList<>();
    synapses.add(new DirectedSynapsesImpl(getPrimaryAxons(), getPrimaryActivationFunction()));
    return synapses;
  }
}
