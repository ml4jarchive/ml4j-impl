package org.ml4j.nn.synapses;

import org.ml4j.MatrixFactory;
import org.ml4j.nn.activationfunctions.DifferentiableActivationFunction;
import org.ml4j.nn.axons.Axons;
import org.ml4j.nn.axons.FullyConnectedAxonsImpl;
import org.ml4j.nn.axons.PassThroughAxonsImpl;
import org.ml4j.nn.graph.DirectedDipoleGraph;
import org.ml4j.nn.graph.DirectedDipoleGraphImpl;
import org.ml4j.nn.graph.DirectedPath;
import org.ml4j.nn.graph.DirectedPathImpl;
import org.ml4j.nn.neurons.Neurons;
import org.ml4j.nn.neurons.NeuronsActivation;

public class ResidualSynapsesImpl<L extends Neurons, R extends Neurons>
    extends DirectedSynapsesImpl<L, R> {
  
  /**
   * Default serialization id.
   */
  private static final long serialVersionUID = 1L;
  private Axons<?, ?, ?> residualAxons;
  private Neurons residualActivationSource;

  /**
   * @param primaryAxons The primary axons.
   * @param activationFunction The activation function.
   */
  public ResidualSynapsesImpl(Axons<? extends L, ? extends R, ?> primaryAxons,
      Neurons residualActivationSource, DifferentiableActivationFunction activationFunction,
      MatrixFactory matrixFactory) {
    super(primaryAxons, activationFunction);
    this.residualActivationSource = residualActivationSource;
    if (residualActivationSource.getNeuronCountExcludingBias() 
        == primaryAxons.getRightNeurons().getNeuronCountExcludingBias()) { 
      this.residualAxons = new PassThroughAxonsImpl(residualActivationSource,
          primaryAxons.getRightNeurons());
    } else {
      this.residualAxons = new FullyConnectedAxonsImpl(residualActivationSource,
      primaryAxons.getRightNeurons(), matrixFactory);
    }   
  }

  protected ResidualSynapsesImpl(Axons<? extends L, ? extends R, ?> primaryAxons,
      Neurons residualActivationSource, DirectedDipoleGraph<Axons<?, ?, ?>> axonsGraph,
      DifferentiableActivationFunction activationFunction, Axons<?, ?, ?> residualAxons) {
    super(primaryAxons, axonsGraph, activationFunction);
    this.residualAxons = residualAxons;
    this.residualActivationSource = residualActivationSource;
  }

  @Override
  public DirectedDipoleGraph<Axons<?, ?, ?>> getAxonsGraph() {
    DirectedDipoleGraph<Axons<?, ?, ?>> axonsGraph = new DirectedDipoleGraphImpl<Axons<?, ?, ?>>();
    DirectedPath<Axons<?, ?, ?>> mainPath = new DirectedPathImpl<Axons<?, ?, ?>>(getPrimaryAxons());
    DirectedPath<Axons<?, ?, ?>> residualPath = new DirectedPathImpl<Axons<?, ?, ?>>(residualAxons);
    axonsGraph.addParallelPath(mainPath);
    axonsGraph.addParallelPath(residualPath);
    return axonsGraph;
  }

  @Override
  protected NeuronsActivation getInputNeuronsActivationForPathIndex(
      DirectedSynapsesInput synapsesInput, int pathIndex) {
    if (pathIndex != 0 && pathIndex != 1) {
      throw new IllegalArgumentException("Path index:" + pathIndex + " not valid for "
          + "ResidualSynapsesImpl - custom classes can override this behaviour");
    }
    return pathIndex == 0 ? synapsesInput.getInput() : synapsesInput.getResidualInput();
  }

  @Override
  public DirectedSynapses<L, R> dup() {
    return new ResidualSynapsesImpl<L, R>(getPrimaryAxons().dup(), residualActivationSource,
        cloneAxonsGraph(), getActivationFunction(), residualAxons.dup());
  }
}
