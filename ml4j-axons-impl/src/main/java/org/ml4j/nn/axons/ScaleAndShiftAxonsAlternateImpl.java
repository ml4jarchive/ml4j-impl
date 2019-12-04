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

package org.ml4j.nn.axons;

import org.ml4j.EditableMatrix;
import org.ml4j.Matrix;
import org.ml4j.MatrixFactory;
import org.ml4j.nn.neurons.ImageNeuronsActivation;
import org.ml4j.nn.neurons.ImageNeuronsActivationImpl;
import org.ml4j.nn.neurons.Neurons;
import org.ml4j.nn.neurons.Neurons3D;
import org.ml4j.nn.neurons.NeuronsActivation;
import org.ml4j.nn.neurons.NeuronsActivationImpl;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Naive sparse-matrix implementation of ScaleAndShiftAxons.
 * 
 * @author Michael Lavelle
 *
 */
public class ScaleAndShiftAxonsAlternateImpl<N extends Neurons>
		extends TrainableAxonsBaseAlternateImpl<N, N, ScaleAndShiftAxons<N>, ScaleAndShiftAxonsConfig>
		implements ScaleAndShiftAxons<N> {

	/**
	 * Default serialization id.
	 */
	private static final long serialVersionUID = 1L;

	private static final Logger LOGGER = LoggerFactory.getLogger(ScaleAndShiftAxonsAlternateImpl.class);

	/**
	 * @param leftNeurons
	 *            The neurons whose activations we want to scale and shift.
	 * @param rightNeurons
	 *            The target neurons.
	 * @param matrixFactory
	 *            The MatrixFactory to use to initialise the weights.
	 * @param config
	 *            The config for these Axons.
	 */
	public ScaleAndShiftAxonsAlternateImpl(N leftNeurons, N rightNeurons, N singleNeuronWithBias,
			MatrixFactory matrixFactory, ScaleAndShiftAxonsConfig config) {
		super(singleNeuronWithBias, rightNeurons, matrixFactory, config);
		if (!leftNeurons.hasBiasUnit()) {
			throw new IllegalArgumentException("Left neurons must contain " + "a bias unit for ScaleAndShiftAxons");
		}
		if (leftNeurons.getNeuronCountExcludingBias() != rightNeurons.getNeuronCountExcludingBias()) {
			throw new IllegalArgumentException("Left neurons and right neurons are not compatible sizes:"
					+ leftNeurons.getNeuronCountExcludingBias() + ":" + rightNeurons.getNeuronCountExcludingBias());
		}
		if (!leftNeurons.hasBiasUnit()) {
			throw new IllegalArgumentException("Left neurons must contain " + "a bias unit for ScaleAndShiftAxons");
		}
	}

	protected ScaleAndShiftAxonsAlternateImpl(N leftNeurons, N rightNeurons, Matrix connectionWeights, Matrix biases,
			ScaleAndShiftAxonsConfig config) {
		super(leftNeurons, rightNeurons, connectionWeights, biases, config);
		if (leftNeurons.getNeuronCountExcludingBias() != rightNeurons.getNeuronCountExcludingBias()) {
			throw new IllegalArgumentException("Left neurons and right neurons are not compatible sizes");
		}
		if (!leftNeurons.hasBiasUnit()) {
			throw new IllegalArgumentException("Left neurons must contain " + "a bias unit for ScaleAndShiftAxons");
		}
	}

	/**
	 * Obtain the initial axon connection weights.
	 * 
	 * @param inputNeurons
	 *            The input Neurons
	 * @param outputNeurons
	 *            The output Neurons
	 * @param matrixFactory
	 *            The matrix factory
	 * @return The initial connection weights
	 */
	protected Matrix createDefaultInitialConnectionWeights(MatrixFactory matrixFactory) {

		LOGGER.debug("Initialising FullyConnectedAxon weights...");

		if (rightNeurons.hasBiasUnit()) {
			throw new IllegalArgumentException("Right neurons should not contain bias unit");
		}

		EditableMatrix weights = (EditableMatrix)matrixFactory.createOnes(rightNeurons.getNeuronCountIncludingBias(), 1);

		weights.putColumn(0, config.getScaleColumnVector().asEditableMatrix());

		return weights;
	}

	/**
	 * Obtain the initial axon connection weights.
	 * 
	 * @param inputNeurons
	 *            The input Neurons
	 * @param outputNeurons
	 *            The output Neurons
	 * @param matrixFactory
	 *            The matrix factory
	 * @return The initial connection weights
	 */
	protected Matrix createDefaultInitialLeftToRightBiases(MatrixFactory matrixFactory) {

		LOGGER.debug("Initialising FullyConnectedAxon weights...");

		if (rightNeurons.hasBiasUnit()) {
			throw new IllegalArgumentException("Right neurons should not contain bias unit");
		}

		EditableMatrix biases = (EditableMatrix)matrixFactory.createZeros(rightNeurons.getNeuronCountIncludingBias(), 1);
		biases.putColumn(0, config.getShiftColumnVector().asEditableMatrix());
		return biases;
	}

	@Override
	public AxonsActivation pushLeftToRight(NeuronsActivation leftNeuronsActivation,
			AxonsActivation previousRightToLeftActivation, AxonsContext axonsContext) {

		// TODO MON - muli
		Matrix result = leftNeuronsActivation.getActivations(axonsContext.getMatrixFactory()).asEditableMatrix().muliColumnVector(getScaleColumnVector()).asEditableMatrix().addiColumnVector(getShiftColumnVector());

		NeuronsActivation output = null;
				if (leftNeuronsActivation instanceof ImageNeuronsActivation) {
					output = new ImageNeuronsActivationImpl(result, (Neurons3D) this.rightNeurons, leftNeuronsActivation.getFeatureOrientation(), false);
				} else {
					output = new NeuronsActivationImpl(result, leftNeuronsActivation.getFeatureOrientation());
				}
		// new NeuronsActivation(result, leftNeuronsActivation.getFeatureOrientation());

		//NeuronsActivation input = new NeuronsActivation(leftNeuronsActivation.getActivations(),
		//		NeuronsActivationFeatureOrientation.ROWS_SPAN_FEATURE_SET);

		return new AxonsActivationImpl(this, null, leftNeuronsActivation, output, leftNeurons, rightNeurons, false);
	}

	@Override
	public AxonsActivation pushRightToLeft(NeuronsActivation rightNeuronsActivation,
			AxonsActivation previousLeftToRightActivation, AxonsContext axonsContext) {

		// Matrix xhat = previousLeftToRightActivation.getOutput().getActivations();
		Matrix dout = rightNeuronsActivation.getActivations(axonsContext.getMatrixFactory());

		// Matrix dgamma = xhat.mul(dout).transpose().rowSums().transpose();

		// Matrix dbeta = dout.transpose().rowSums().transpose();

		Matrix nonBiasInputs = dout.mulColumnVector(getScaleColumnVector());
		// Matrix biasInputs = dout.rowSums();
		// Matrix result = biasInputs.appendHorizontally(nonBiasInputs);

		NeuronsActivation output = new NeuronsActivationImpl(nonBiasInputs,
				rightNeuronsActivation.getFeatureOrientation());

		Matrix inputDropoutMask = null;
		return new AxonsActivationImpl(this, inputDropoutMask, rightNeuronsActivation, output, leftNeurons,
				rightNeurons, true);

	}

	@Override
	public ScaleAndShiftAxons<N> dup() {
		// TODO ML
		return new ScaleAndShiftAxonsAlternateImpl<>(leftNeurons, rightNeurons, this.axonWeights.getConnectionWeights(),
				this.axonWeights.getLeftToRightBiases(), config);
	}

	@Override
	public Matrix getScaleColumnVector() {
		return this.axonWeights.getConnectionWeights();
	}

	@Override
	public Matrix getShiftColumnVector() {
		return this.axonWeights.getLeftToRightBiases();
	}

	@Override
	protected Matrix createDefaultInitialRightToLeftBiases(MatrixFactory matrixFactory) {
		throw new UnsupportedOperationException();
	}
}
