package org.ml4j.nn.layers;

import org.ml4j.Matrix;
import org.ml4j.nn.neurons.NeuronsActivation;
import org.ml4j.nn.synapses.DirectedSynapsesActivation;
import org.ml4j.nn.synapses.DirectedSynapsesContext;
import org.ml4j.nn.synapses.DirectedSynapsesGradient;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.List;

public class ResidualBlockLayerActivationImpl extends DirectedLayerActivationImpl {

  private static final Logger LOGGER = LoggerFactory.getLogger(DirectedLayerActivationImpl.class);
  
  public ResidualBlockLayerActivationImpl(DirectedLayer<?, ?> layer,
      List<DirectedSynapsesActivation> synapseActivations, NeuronsActivation outputActivation) {
    super(layer, synapseActivations, outputActivation);
  }

  @Override
  protected DirectedLayerGradient backPropagateAndAddToSynapseGradientList(
      List<DirectedSynapsesGradient> synapseGradientList,
      DirectedSynapsesGradient outerSynapsesGradient,
      List<DirectedSynapsesActivation> activationsToBackPropagateThrough,
      DirectedLayerContext layerContext) {
    
    int index = activationsToBackPropagateThrough.size() - 1;
    NeuronsActivation finalGrad = null;
    NeuronsActivation residualOutput = null;
    DirectedSynapsesGradient synapsesGradient = outerSynapsesGradient;
    residualOutput = synapsesGradient.getResidualOutput();

    for (DirectedSynapsesActivation synapsesActivation : activationsToBackPropagateThrough) {
      
      DirectedSynapsesContext context = layerContext.getSynapsesContext(index);

      synapsesGradient =
          synapsesActivation.backPropagate(synapsesGradient, context);
      
      if (synapsesGradient.getResidualOutput() != null && residualOutput != null) {
        throw new IllegalStateException("Multiple synapses in chain have residual output gradient");
      }
      if (residualOutput == null) {
        residualOutput = synapsesGradient.getResidualOutput();
      }

      synapseGradientList.add(synapsesGradient);
      finalGrad = synapsesGradient.getOutput();
      index--;
    }
    
    if (residualOutput != null) {
      LOGGER.debug("RESIDUAL GRADIENT: " + residualOutput.getActivations().getRows() 
          + ":" + residualOutput.getActivations().getColumns());
      LOGGER.debug("FINAL GRADIENT: " + finalGrad.getActivations().getRows() 
          + ":" + finalGrad.getActivations().getColumns());
      Matrix summedGradient = finalGrad.getActivations().add(residualOutput.getActivations());
      finalGrad = new NeuronsActivation(summedGradient, finalGrad.getFeatureOrientation());
    }

    return new DirectedLayerGradientImpl(finalGrad, synapseGradientList);
  }

}
