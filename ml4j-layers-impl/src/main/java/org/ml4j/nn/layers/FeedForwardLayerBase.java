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
import org.ml4j.nn.axons.ScaleAndShiftAxonsConfig;
import org.ml4j.nn.axons.ScaleAndShiftAxonsImpl;
import org.ml4j.nn.neurons.Neurons;
import org.ml4j.nn.neurons.NeuronsActivation;
import org.ml4j.nn.neurons.NeuronsActivationFeatureOrientation;
import org.ml4j.nn.synapses.ActivationFunctionOnlyDirectedSynapsesImpl;
import org.ml4j.nn.synapses.AxonsOnlyDirectedSynapsesImpl;
import org.ml4j.nn.synapses.BatchNormDirectedSynapsesImpl;
import org.ml4j.nn.synapses.DirectedSynapses;
import org.ml4j.nn.synapses.DirectedSynapsesActivation;
import org.ml4j.nn.synapses.DirectedSynapsesImpl;
import org.ml4j.nn.synapses.DirectedSynapsesInput;
import org.ml4j.nn.synapses.DirectedSynapsesInputImpl;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.List;

/**
 * A default base implementation of FeedForwardLayer.
 * 
 * @author Michael Lavelle
 * 
 * @param <A> The type of primary Axons in this FeedForwardLayer.
 */
public abstract class FeedForwardLayerBase<A extends Axons<?, ?, ?>, 
    L extends FeedForwardLayer<A, L>> 
    implements FeedForwardLayer<A, L> {

  /**
   * Default serialization id.
   */
  private static final long serialVersionUID = 1L;
 
  private static final Logger LOGGER = 
      LoggerFactory.getLogger(FeedForwardLayerBase.class);

  protected A primaryAxons;
  
  protected DifferentiableActivationFunction primaryActivationFunction;
  
  protected MatrixFactory matrixFactory;
  
  protected boolean withBatchNorm;
 
  /**
   * @param primaryAxons The primary Axons
   * @param activationFunction The primary activation function
   * @param matrixFactory The matrix factory
   * @param withBatchNorm Whether to enable batch norm.
   */
  protected FeedForwardLayerBase(A primaryAxons, 
      DifferentiableActivationFunction activationFunction, MatrixFactory matrixFactory, 
      boolean withBatchNorm) {
    this.primaryAxons = primaryAxons;
    this.primaryActivationFunction = activationFunction;
    this.matrixFactory = matrixFactory;
    this.withBatchNorm = withBatchNorm;
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
  public A getPrimaryAxons() {
    return primaryAxons;
  }

  @Override
  public NeuronsActivation getOptimalInputForOutputNeuron(int outputNeuronIndex,
      DirectedLayerContext directedLayerContext) {
    LOGGER.debug("Obtaining optimal input for output neuron with index:" + outputNeuronIndex);
    //int countJ = getPrimaryAxons().getLeftNeurons().getNeuronCountExcludingBias();
    Matrix weights = getPrimaryAxons().getDetachedConnectionWeights();
    int countJ = weights.getRows() - (getPrimaryAxons().getLeftNeurons().hasBiasUnit() ? 1 : 0);
    double[] maximisingInputFeatures = new double[countJ];
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
    LOGGER.debug(directedLayerContext.toString() + ":Forward propagating through layer");
    
    NeuronsActivation inFlightNeuronsActivation = inputNeuronsActivation;
    List<DirectedSynapsesActivation> synapseActivations = new ArrayList<>();
    int synapsesIndex = 0;
    for (DirectedSynapses<?, ?> synapses : getSynapses()) {
      DirectedSynapsesInput input = new DirectedSynapsesInputImpl(inFlightNeuronsActivation);
      DirectedSynapsesActivation inFlightNeuronsSynapseActivation = 
          synapses.forwardPropagate(input, 
              directedLayerContext.createSynapsesContext(synapsesIndex++));
      synapseActivations.add(inFlightNeuronsSynapseActivation);
      inFlightNeuronsActivation = inFlightNeuronsSynapseActivation.getOutput();
    }
 
    return new DirectedLayerActivationImpl(this, synapseActivations, 
        inFlightNeuronsActivation);
  }

  @Override
  public List<DirectedSynapses<?, ?>> getSynapses() {
    List<DirectedSynapses<?, ?>> synapses = new ArrayList<>();
    if (withBatchNorm) {
      
      Matrix initialGamma = matrixFactory.createOnes(1, 
          getPrimaryAxons().getRightNeurons().getNeuronCountExcludingBias());
      Matrix initialBeta = matrixFactory.createZeros(1, 
          getPrimaryAxons().getRightNeurons().getNeuronCountExcludingBias());
      ScaleAndShiftAxonsConfig config = 
          new ScaleAndShiftAxonsConfig(initialGamma, initialBeta);
      
      synapses.add(new AxonsOnlyDirectedSynapsesImpl<Neurons, Neurons>(getPrimaryAxons()));
      
      synapses.add(new BatchNormDirectedSynapsesImpl<Neurons, Neurons>(
          getPrimaryAxons().getRightNeurons(), getPrimaryAxons().getRightNeurons(), 
          new ScaleAndShiftAxonsImpl(getPrimaryAxons().getRightNeurons(), matrixFactory, config)));

      synapses.add(new ActivationFunctionOnlyDirectedSynapsesImpl<Neurons, Neurons>(
          getPrimaryAxons().getRightNeurons(), getPrimaryAxons().getRightNeurons(),
          getPrimaryActivationFunction()));
      
    } else {
      synapses.add(new DirectedSynapsesImpl<Neurons, Neurons>(
          getPrimaryAxons(), getPrimaryActivationFunction()));
    }
    return synapses;
  }
}
