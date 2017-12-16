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

package org.ml4j.nn.synapses;

import org.ml4j.Matrix;
import org.ml4j.nn.activationfunctions.DifferentiableActivationFunction;
import org.ml4j.nn.activationfunctions.DifferentiableActivationFunctionActivation;
import org.ml4j.nn.axons.Axons;
import org.ml4j.nn.axons.AxonsActivation;
import org.ml4j.nn.graph.DirectedDipoleGraph;
import org.ml4j.nn.graph.DirectedDipoleGraphImpl;
import org.ml4j.nn.graph.DirectedPath;
import org.ml4j.nn.graph.DirectedPathImpl;
import org.ml4j.nn.neurons.Neurons;
import org.ml4j.nn.neurons.NeuronsActivation;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Default implementation of DirectedSynapses.
 * 
 * @author Michael Lavelle
 */
public class DirectedSynapsesImpl<L extends Neurons, R extends Neurons> 
    implements DirectedSynapses<L, R> {

  /**
   * Default serialization id.
   */
  private static final long serialVersionUID = 1L;
  
  private static final Logger LOGGER = 
      LoggerFactory.getLogger(DirectedSynapsesImpl.class);
  
  private Axons<? extends L, ? extends R, ?> primaryAxons;
  private DifferentiableActivationFunction activationFunction;
  private DirectedDipoleGraph<Axons<?, ?, ?>> axonsGraph;
  
  /**
   * Create a new implementation of DirectedSynapses.
   * 
   * @param primaryAxons The primary Axons within these synapses
   * @param axonsGraph The axons graph within these Synapses.
   * @param activationFunction The activation function within these synapses
   */
  protected DirectedSynapsesImpl(Axons<? extends L, ? extends R, ?> primaryAxons, 
      DirectedDipoleGraph<Axons<?, ?, ?>> axonsGraph, 
      DifferentiableActivationFunction activationFunction) {
    super();
    this.primaryAxons = primaryAxons;
    this.activationFunction = activationFunction;
    this.axonsGraph = axonsGraph;
  }
  
  /**
   * Create a new implementation of DirectedSynapses.
   * 
   * @param primaryAxons The primary Axons within these synapses
   * @param activationFunction The activation function within these synapses
   */
  public DirectedSynapsesImpl(Axons<? extends L, ? extends R, ?> primaryAxons, 
      DifferentiableActivationFunction activationFunction) {
      this(primaryAxons, new DirectedDipoleGraphImpl<Axons<?, ? ,?>>(primaryAxons), 
          activationFunction);
  }

  @Override
  public Axons<? extends L, ? extends R, ?> getPrimaryAxons() {
    return primaryAxons;
  }
  
  /**
   * @return The Axons graph within these DirectedSynapses.
   */
  public DirectedDipoleGraph<Axons<?, ?, ?>> getAxonsGraph() {
    return new DirectedDipoleGraphImpl<Axons<?, ?, ?>>(primaryAxons);
  }

  @Override
  public DirectedSynapses<L, R> dup() {
    return new DirectedSynapsesImpl<L, R>(primaryAxons.dup(), cloneAxonsGraph() , 
        activationFunction);
  }
  
  private DirectedDipoleGraph<Axons<?, ?, ?>> cloneAxonsGraph() {

    DirectedDipoleGraph<Axons<?, ?, ?>> dup = new DirectedDipoleGraphImpl<Axons<?, ?, ?>>();
    for (DirectedPath<Axons<?, ?, ?>> directedPath : axonsGraph.getParallelPaths()) {
      DirectedPath<Axons<?, ?, ?>> dupPath = new DirectedPathImpl<Axons<?, ?, ?>>();
      for (Axons<?, ?, ?> axons : directedPath.getEdges()) {
        Axons<?, ?, ?> dupAxons = axons.dup();
        dupPath.addEdge(dupAxons);
      }
      dup.addParallelPath(dupPath);
    }
    return dup;
  }

  @Override
  public DifferentiableActivationFunction getActivationFunction() {
    return activationFunction;
  }


  @Override
  public DirectedSynapsesActivation forwardPropagate(DirectedSynapsesInput input,
      DirectedSynapsesContext synapsesContext) {

    LOGGER.debug("Forward propagating through DirectedSynapses");

    NeuronsActivation inputNeuronsActivation = input.getInput();

    Matrix totalAxonsOutputMatrix = null;
    
    NeuronsActivation axonsOutputActivation = null;

    DirectedDipoleGraph<AxonsActivation> axonsActivationGraph =
        new DirectedDipoleGraphImpl<AxonsActivation>();

    int pathIndex = 0;
    
    for (DirectedPath<Axons<?, ?, ?>> parallelAxonsPath : this.getAxonsGraph().getParallelPaths()) {

      DirectedPath<AxonsActivation> axonsActivationPath = new DirectedPathImpl<AxonsActivation>();

      int axonsIndex = 0;
      
      for (Axons<?, ?, ?> axons : parallelAxonsPath.getEdges()) {

        AxonsActivation axonsActivation =
            axons.pushLeftToRight(inputNeuronsActivation, null, 
                synapsesContext.getAxonsContext(pathIndex, axonsIndex));

        axonsActivationPath.addEdge(axonsActivation);
        axonsOutputActivation = axonsActivation.getOutput();
        inputNeuronsActivation = axonsOutputActivation;
        axonsIndex++;
      }
      if (totalAxonsOutputMatrix == null) {
        totalAxonsOutputMatrix = inputNeuronsActivation.getActivations();
      } else {
        
        Matrix axonsPathOutputActivationMatrix = axonsOutputActivation.getActivations();
        
        if (axonsPathOutputActivationMatrix.getRows() != totalAxonsOutputMatrix.getRows()) {
          throw new IllegalStateException(
              "Final axons activation in each parallel path must be the "
              + "same dimensions");
        }
        if (axonsPathOutputActivationMatrix.getColumns() 
            != totalAxonsOutputMatrix.getColumns()) {
          throw new IllegalStateException(
              "Final axons activation in each parallel path must be the " + "same dimensions");
        }
        
        totalAxonsOutputMatrix =
            totalAxonsOutputMatrix.add(axonsPathOutputActivationMatrix);
      }
      axonsActivationGraph.addParallelPath(axonsActivationPath);
      pathIndex++;
    }
    
    DifferentiableActivationFunctionActivation activationFunctionActivation =
        activationFunction.activate(axonsOutputActivation, synapsesContext);

    NeuronsActivation outputNeuronsActivation = activationFunctionActivation.getOutput();

    return new DirectedSynapsesActivationImpl(this, inputNeuronsActivation, axonsActivationGraph,
        activationFunctionActivation, outputNeuronsActivation);

  }

  @Override
  public L getLeftNeurons() {
    return primaryAxons.getLeftNeurons();
  }

  @Override
  public R getRightNeurons() {
    return primaryAxons.getRightNeurons();
  }
}
