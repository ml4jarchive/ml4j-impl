package org.ml4j.nn.layers;

import org.ml4j.MatrixFactory;
import org.ml4j.nn.activationfunctions.DifferentiableActivationFunction;
import org.ml4j.nn.axons.Axons;
import org.ml4j.nn.axons.FullyConnectedAxonsImpl;
import org.ml4j.nn.axons.PassThroughAxonsImpl;
import org.ml4j.nn.axons.TrainableAxons;
import org.ml4j.nn.neurons.Neurons;
import org.ml4j.nn.neurons.NeuronsActivation;
import org.ml4j.nn.synapses.DirectedSynapses;
import org.ml4j.nn.synapses.DirectedSynapsesActivation;
import org.ml4j.nn.synapses.DirectedSynapsesContext;
import org.ml4j.nn.synapses.DirectedSynapsesInput;
import org.ml4j.nn.synapses.DirectedSynapsesInputImpl;
import org.ml4j.nn.synapses.ResidualSynapsesImpl;

import java.util.ArrayList;
import java.util.List;

public class ResidualBlockLayerImpl<A extends TrainableAxons<?, ?, A>, 
    L extends FeedForwardLayer<A, L>>
    implements FeedForwardLayer<A, ResidualBlockLayerImpl<A, L>> {

  /**
   * Default serialization id.
   */
  private static final long serialVersionUID = 1L;

  private L layer1;
  private L layer2;
  private MatrixFactory matrixFactory;
  private Axons<?, ?, ?> residualAxons;

  /**
   * @param layer1 The first Layer in this Residual Block.
   * @param layer2 The second Layer in this Residual Block.
   * @param matrixFactory The MatrixFactory.
   */
  public ResidualBlockLayerImpl(L layer1, L layer2, MatrixFactory matrixFactory) {
    this.layer1 = layer1;
    this.layer2 = layer2;
    this.matrixFactory = matrixFactory;
    
    Neurons residualActivationSource = 
        new Neurons(layer1.getSynapses().get(0).getLeftNeurons()
            .getNeuronCountExcludingBias(), false);
    DirectedSynapses<?, ?> firstSynapsesOfSecondLayer = layer2.getSynapses().get(0);

    Axons<?, ?, ?> residualSynapsesPrimaryAxons = firstSynapsesOfSecondLayer.getPrimaryAxons();
    
    if (residualActivationSource.getNeuronCountExcludingBias() 
        == residualSynapsesPrimaryAxons.getRightNeurons().getNeuronCountExcludingBias()) { 
      residualAxons = new PassThroughAxonsImpl(residualActivationSource,
          residualSynapsesPrimaryAxons.getRightNeurons());
    } else {
      residualAxons = new FullyConnectedAxonsImpl(residualActivationSource,
          residualSynapsesPrimaryAxons.getRightNeurons(), matrixFactory);
    }   
  }
  
  /**
   * @param layer1 The first Layer in this Residual Block.
   * @param layer2 The second Layer in this Residual Block.
   * @param matrixFactory The MatrixFactory.
   * @param residualAxons The residualAxons.
   */
  protected ResidualBlockLayerImpl(L layer1, L layer2, MatrixFactory matrixFactory, 
      Axons<?, ?, ?> residualAxons) {
    this.layer1 = layer1;
    this.layer2 = layer2;
    this.matrixFactory = matrixFactory;
    this.residualAxons = residualAxons;
  }

  @Override
  public DirectedLayerActivation forwardPropagate(NeuronsActivation inputNeuronsActivation,
      DirectedLayerContext directedLayerContext) {

    NeuronsActivation inFlightNeuronsActivation = inputNeuronsActivation;
    List<DirectedSynapsesActivation> synapseActivations = new ArrayList<>();
    int secondLayerInputSynapsesIndex = layer1.getSynapses().size();
    int synapsesIndex = 0;
    for (DirectedSynapses<?, ?> synapses : getSynapses()) {
      NeuronsActivation residualInput =
          (synapsesIndex == secondLayerInputSynapsesIndex) ? inputNeuronsActivation : null;
      DirectedSynapsesInput input =
          new DirectedSynapsesInputImpl(inFlightNeuronsActivation, residualInput);
      DirectedSynapsesActivation inFlightNeuronsSynapseActivation = synapses.forwardPropagate(input,
          directedLayerContext.getSynapsesContext(synapsesIndex++));
      synapseActivations.add(inFlightNeuronsSynapseActivation);
      inFlightNeuronsActivation = inFlightNeuronsSynapseActivation.getOutput();
    }

    return new ResidualBlockLayerActivationImpl(this, synapseActivations,
        inFlightNeuronsActivation);
  }

  @Override
  public int getInputNeuronCount() {
    return layer1.getInputNeuronCount();
  }

  @Override
  public NeuronsActivation getOptimalInputForOutputNeuron(int outputNeuronsIndex,
      DirectedLayerContext layerContext) {
    throw new UnsupportedOperationException("Not supported for ResidualBlockLayers");
  }



  public L getFirstLayer() {
    return layer1;
  }

  public FeedForwardLayer<?, ?> getSecondLayer() {
    return layer2;
  }

  @Override
  public int getOutputNeuronCount() {
    return layer2.getOutputNeuronCount();
  }

  @Override
  public DifferentiableActivationFunction getPrimaryActivationFunction() {
    throw new UnsupportedOperationException("Not supported for ResidualBlockLayers");
  }

  @Override
  public A getPrimaryAxons() {
    throw new UnsupportedOperationException("Not supported for ResidualBlockLayers");
  }

  @Override
  public List<DirectedSynapses<?, ?>> getSynapses() {

    List<DirectedSynapses<?, ?>> synapses = new ArrayList<DirectedSynapses<?, ?>>();
    synapses.addAll(layer1.getSynapses());
    List<DirectedSynapses<?, ?>> remainingSynapses =
        layer2.getSynapses().subList(1, layer2.getSynapses().size());
    
    Neurons residualActivationSource = 
        new Neurons(layer1.getSynapses().get(0).getLeftNeurons()
            .getNeuronCountExcludingBias(), false);
    DirectedSynapses<?, ?> firstSynapsesOfSecondLayer = layer2.getSynapses().get(0);

    Axons<?, ?, ?> residualSynapsesPrimaryAxons = firstSynapsesOfSecondLayer.getPrimaryAxons();
   
    synapses.add(new ResidualSynapsesImpl<Neurons, Neurons>(residualSynapsesPrimaryAxons , 
        residualAxons, residualActivationSource,
        firstSynapsesOfSecondLayer.getActivationFunction(), matrixFactory));
    synapses.addAll(remainingSynapses);
    return synapses;
  }

  @Override
  public ResidualBlockLayerImpl<A, L> dup() {
    return new ResidualBlockLayerImpl<A, L>(layer1.dup(), layer2.dup(), matrixFactory, 
        residualAxons.dup());
  }
  
  @Override
  public double getTotalRegularisationCost(DirectedLayerContext layerContext) {
    double totalRegularisationCost = 0d;
    int synapsesIndex = 0;
    for (DirectedSynapses<?, ?> synapses : getSynapses()) {
      DirectedSynapsesContext synapsesContext = layerContext.getSynapsesContext(synapsesIndex);
      totalRegularisationCost =
          totalRegularisationCost + synapses.getTotalRegularisationCost(synapsesContext);
      synapsesIndex++;
    }
    return totalRegularisationCost;
  }
}
