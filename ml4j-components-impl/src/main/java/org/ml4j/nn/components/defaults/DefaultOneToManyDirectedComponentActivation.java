package org.ml4j.nn.components.defaults;

import java.util.List;

import org.ml4j.MatrixFactory;
import org.ml4j.nn.components.OneToManyDirectedComponentActivation;
import org.ml4j.nn.neurons.ImageNeuronsActivation;
import org.ml4j.nn.neurons.ImageNeuronsActivationImpl;
import org.ml4j.nn.neurons.Neurons3D;
import org.ml4j.nn.neurons.NeuronsActivation;
import org.ml4j.nn.neurons.NeuronsActivationImpl;

public class DefaultOneToManyDirectedComponentActivation extends OneToManyDirectedComponentActivation<NeuronsActivation> {

	private MatrixFactory matrixFactory;
	
	public DefaultOneToManyDirectedComponentActivation(List<NeuronsActivation> activations, MatrixFactory matrixFactory) {
		super(activations);
		this.matrixFactory = matrixFactory;
	}

	@Override
	protected NeuronsActivation getBackPropagatedGradient(List<NeuronsActivation> gradient) {
		boolean allImage = (gradient.stream().allMatch(g -> g instanceof ImageNeuronsActivation));
		NeuronsActivation totalActivation = null;
		if (allImage) {
			totalActivation = new NeuronsActivationImpl(matrixFactory.createMatrix(gradient.get(0).getRows(), 
					gradient.get(0).getColumns()), gradient.get(0).getFeatureOrientation(), false);
		} else {
			if (true) {
				for (NeuronsActivation a : gradient) {
					if (!(a instanceof ImageNeuronsActivation)) {
						System.out.println(a.getStackTrace());
					}
				}

				throw new UnsupportedOperationException();
			}
			totalActivation = new NeuronsActivationImpl(matrixFactory.createMatrix(gradient.get(0).getActivations(matrixFactory).getRows(), 
					gradient.get(0).getActivations(matrixFactory).getColumns()), gradient.get(0).getFeatureOrientation(), false);
		}
		for (NeuronsActivation activation : gradient) {
			totalActivation.addInline(matrixFactory, activation);
		}
		if (allImage) {
			 return new ImageNeuronsActivationImpl(totalActivation.getActivations(matrixFactory), (Neurons3D)gradient.get(0).getNeurons(), gradient.get(0).getFeatureOrientation(), false);
		} else {
			return totalActivation;
		}

	}
}
