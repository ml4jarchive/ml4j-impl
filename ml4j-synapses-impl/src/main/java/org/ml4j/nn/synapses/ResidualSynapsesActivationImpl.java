package org.ml4j.nn.synapses;

import org.ml4j.nn.activationfunctions.DifferentiableActivationFunctionActivation;
import org.ml4j.nn.axons.AxonsActivation;
import org.ml4j.nn.costfunctions.CostFunctionGradient;
import org.ml4j.nn.graph.DirectedDipoleGraph;
import org.ml4j.nn.neurons.NeuronsActivation;

public class ResidualSynapsesActivationImpl extends DirectedSynapsesActivationImpl {

  public ResidualSynapsesActivationImpl(DirectedSynapses<?, ?> synapses,
      DirectedSynapsesInput inputActivation, DirectedDipoleGraph<AxonsActivation> axonsActivation,
      DifferentiableActivationFunctionActivation activationFunctionActivation,
      NeuronsActivation outputActivation) {
    super(synapses, inputActivation, axonsActivation, activationFunctionActivation,
        outputActivation);
  }

  @Override
  public DirectedSynapsesGradient backPropagate(DirectedSynapsesGradient da,
      DirectedSynapsesContext context) {
    return super.backPropagate(da, context);
  }

  @Override
  public DirectedSynapsesGradient backPropagate(CostFunctionGradient da,
      DirectedSynapsesContext context) {
    return super.backPropagate(da, context);
  }



}
