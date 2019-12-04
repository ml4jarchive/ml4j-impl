package org.ml4j.nn.axons;

import org.ml4j.EditableMatrix;
import org.ml4j.Matrix;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class AxonWeightsImpl implements AxonWeights {
	
	/**
	 * Default serialization id.
	 */
	private static final long serialVersionUID = 1L;

	private static final Logger LOGGER = LoggerFactory.getLogger(AxonWeightsImpl.class);
	
	private EditableMatrix leftToRightBiases;
	private EditableMatrix rightToLeftBiases;
	private EditableMatrix connectionWeights;
	//private Matrix connectionWeightDeltas;

	private int inputNeuronCount;
	private int outputNeuronCount;
	private boolean biases;
	private int connectionWeightsRows;
	private int connectionWeightsColumns;
	private int biasesRows;
	private int biasesColumns;
	private Matrix previousResultForInput;
	private Matrix previousResultForGradient;
	

	
	public AxonWeightsImpl(int inputNeuronCount, int outputNeuronCount, Matrix connectionWeights, Matrix leftToRightBiases, Matrix rightToLeftBiases) {
		this.inputNeuronCount = inputNeuronCount;
		this.outputNeuronCount = outputNeuronCount;
		this.connectionWeights = connectionWeights.asEditableMatrix();
		if (leftToRightBiases != null) {
			this.leftToRightBiases = leftToRightBiases.asEditableMatrix();
			this.biasesRows = leftToRightBiases.getRows();
			this.biasesColumns = leftToRightBiases.getColumns();
		}
		if (rightToLeftBiases != null) {
			this.rightToLeftBiases = rightToLeftBiases.asEditableMatrix();
		}
		this.connectionWeightsRows = connectionWeights.getRows();
		this.connectionWeightsColumns = connectionWeights.getColumns();
		//this.connectionWeightDeltas = new JBlasMatrixFactory2().createMatrix(connectionWeights.getRows(), connectionWeights.getColumns());

		this.biases = leftToRightBiases != null;
	}

	@Override
	public Matrix getConnectionWeights() {
		return connectionWeights;
	}

	@Override
	public Matrix getLeftToRightBiases() {
		return leftToRightBiases;
	}

	@Override
	public Matrix getRightToLeftBiases() {
		return rightToLeftBiases;
	}
	
	/*
	public Matrix getConnectionWeightDeltas() {
		return connectionWeightDeltas;
	}
	*/

	@Override
	public void adjustLeftToRightBiases(Matrix adjustment, ConnectionWeightsAdjustmentDirection adjustmentDirection) {
		  if (adjustmentDirection == ConnectionWeightsAdjustmentDirection.ADDITION) {
			  LOGGER.debug("Adding adjustment to connection weights");
			  leftToRightBiases.addi(adjustment);
		  } else {
			  LOGGER.debug("Subtracting adjustment from connection weights");
			  leftToRightBiases.subi(adjustment);
		  }
	}

	@Override
	public void adjustRightToLeftBiases(Matrix adjustment, ConnectionWeightsAdjustmentDirection adjustmentDirection) {
		 if (adjustmentDirection == ConnectionWeightsAdjustmentDirection.ADDITION) {
			  LOGGER.debug("Adding adjustment to connection weights");
			  rightToLeftBiases.addi(adjustment);
		  } else {
			  LOGGER.debug("Subtracting adjustment from connection weights");
			  rightToLeftBiases.subi(adjustment);
		  }		
	}

	@Override
	public void adjustConnectionWeights(Matrix adjustment, ConnectionWeightsAdjustmentDirection adjustmentDirection,
			boolean initialisation) {
		 if (adjustmentDirection == ConnectionWeightsAdjustmentDirection.ADDITION) {
			  LOGGER.debug("Adding adjustment to connection weights");
			  connectionWeights.addi(adjustment);
		  } else {
			  LOGGER.debug("Subtracting adjustment from connection weights");
			  connectionWeights.subi(adjustment);
		  }		
	}

	@Override
	public int getInputNeuronCount() {
		return inputNeuronCount;
	}

	@Override
	public int getOutputNeuronsCount() {
		return outputNeuronCount;
	}

	@Override
	public Matrix applyToInput(Matrix input) {
		return connectionWeights.mmul(input);
	}

	@Override
	public Matrix applyToGradient(Matrix input) {
		return connectionWeights.transpose().mmul(input);
	}

}
