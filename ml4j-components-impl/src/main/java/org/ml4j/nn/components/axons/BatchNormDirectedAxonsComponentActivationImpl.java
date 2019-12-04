package org.ml4j.nn.components.axons;

import java.util.Arrays;
import java.util.List;

import org.ml4j.EditableMatrix;
import org.ml4j.InterrimMatrix;
import org.ml4j.Matrix;
import org.ml4j.nn.axons.AxonsActivation;
import org.ml4j.nn.axons.AxonsContext;
import org.ml4j.nn.axons.AxonsGradientImpl;
import org.ml4j.nn.axons.ScaleAndShiftAxons;
import org.ml4j.nn.axons.TrainableAxons;
import org.ml4j.nn.components.ChainableDirectedComponentActivation;
import org.ml4j.nn.components.DirectedComponentGradient;
import org.ml4j.nn.components.DirectedComponentGradientImpl;
import org.ml4j.nn.neurons.NeuronsActivation;
import org.ml4j.nn.neurons.NeuronsActivationImpl;

public class BatchNormDirectedAxonsComponentActivationImpl implements DirectedAxonsComponentActivation {

	private ScaleAndShiftAxons<?> scaleAndShiftAxons;
	private AxonsActivation scaleAndShiftAxonsActivation;
	private Matrix meanColumnVector;
	private Matrix varianceColumnVector;
	private BatchNormDirectedAxonsComponent<?, ?> batchNormAxons;
	private AxonsContext axonsContext;

	/**
	 * @param synapses
	 *            The synapses.
	 * @param scaleAndShiftAxons
	 *            The scale and shift axons.
	 * @param inputActivation
	 *            The input activation.
	 * @param axonsActivation
	 *            The axons activation.
	 * @param activationFunctionActivation
	 *            The activation function activation.
	 * @param outputActivation
	 *            The output activation.
	 */
	public BatchNormDirectedAxonsComponentActivationImpl(BatchNormDirectedAxonsComponent<?, ?> batchNormAxons,
			ScaleAndShiftAxons<?> scaleAndShiftAxons, AxonsActivation scaleAndShiftAxonsActivation,
			Matrix meanColumnVector, Matrix varianceColumnVector, AxonsContext axonsContext) {
		this.batchNormAxons = batchNormAxons;
		this.scaleAndShiftAxons = scaleAndShiftAxons;
		this.scaleAndShiftAxonsActivation = scaleAndShiftAxonsActivation;
		this.axonsContext = axonsContext;
		this.meanColumnVector = meanColumnVector;
		this.varianceColumnVector = varianceColumnVector;
	}
	
	private Matrix getStdDevColumnVector(Matrix varianceColumnVector) {
		EditableMatrix stdDev = varianceColumnVector.dup().asEditableMatrix();
		float epsilion = 0.01f;
		for (int i = 0; i < stdDev.getLength(); i++) {
			float variance = stdDev.get(i);
			float stdDevValue = (float)Math.sqrt(variance + epsilion);
			stdDev.put(i, stdDevValue);
		}
		return stdDev;
	}

	@Override
	public DirectedComponentGradient<NeuronsActivation> backPropagate(
			DirectedComponentGradient<NeuronsActivation> outerGradient) {

		// Build up the exponentially weighted averages

		Matrix exponentiallyWeightedAverageMean = batchNormAxons.getExponentiallyWeightedAverageInputFeatureMeans();

		float beta = batchNormAxons.getBetaForExponentiallyWeightedAverages();

		try (InterrimMatrix varianceColumnVectorMul1MinusBeta = varianceColumnVector.mul(1 - (float)beta).asInterrimMatrix()) {
			
			try (InterrimMatrix meanColumnVectorMul1MinusBeta = meanColumnVector.mul(1 - (float)beta).asInterrimMatrix()) {
				
			if (exponentiallyWeightedAverageMean == null) {
				exponentiallyWeightedAverageMean = meanColumnVector.dup();
			} else {
				exponentiallyWeightedAverageMean.asEditableMatrix().muli(beta)
						.addi(meanColumnVectorMul1MinusBeta);
			}
			batchNormAxons.setExponentiallyWeightedAverageInputFeatureMeans(exponentiallyWeightedAverageMean);
	
			Matrix exponentiallyWeightedAverageVariance = batchNormAxons
					.getExponentiallyWeightedAverageInputFeatureVariances();
	
			if (exponentiallyWeightedAverageVariance == null) {
				exponentiallyWeightedAverageVariance = varianceColumnVector.dup();
			} else {
				exponentiallyWeightedAverageVariance.asEditableMatrix().muli(beta)
						.addiColumnVector(varianceColumnVectorMul1MinusBeta);
			}
			batchNormAxons.setExponentiallyWeightedAverageInputFeatureVariances(exponentiallyWeightedAverageVariance);
			}
		}
		
		Matrix xhat = scaleAndShiftAxonsActivation.getPostDropoutInput().getActivations(axonsContext.getMatrixFactory());
		Matrix dout = outerGradient.getOutput().getActivations(axonsContext.getMatrixFactory());

		/**
		 * . xhat:1000:101COLUMNS_SPAN_FEATURE_SET dout:100:1000ROWS_SPAN_FEATURE_SET
		 * 
		 * 
		 * 
		 */

		// System.out.println(
		// "xhat:" + xhat.getRows() + ":" + xhat.getColumns() +
		// xhatn.getFeatureOrientation());
		// Matrix dbeta = outerGradient.
		// System.out.println(
		// "dout:" + dout.getRows() + ":" + dout.getColumns()
		// + outerGradient.getFeatureOrientation());

		try (InterrimMatrix xhatMulDout = xhat.mul(dout).asInterrimMatrix()) {
		
		Matrix dgammaColumnVector = xhatMulDout.rowSums();
		
		if (axonsContext.getRegularisationLambda() != 0) {

			//LOGGER.debug("Calculating total regularisation Gradients");

			try (InterrimMatrix connectionWeightsCopy = scaleAndShiftAxons.getScaleColumnVector().asInterrimMatrix()) {
				Matrix regularisationAddition1 = connectionWeightsCopy.asEditableMatrix().muli(axonsContext.getRegularisationLambda());

				dgammaColumnVector.asEditableMatrix()
						.addi(regularisationAddition1);
				
		
			}
		}

		// gamma, xhat, istd = cache
		// N, _ = dout.shape

		// dbeta = np.sum(dout, axis=0)
		// dgamma = np.sum(xhat * dout, axis=0)
		// dx = (gamma*istd/N) * (N*dout - xhat*dgamma - dbeta)

		// return dx, dgamma, dbeta

		// System.out.println("dgamma:" + dgamma.getRows() + ":" + dgamma.getColumns());
		// System.out.println("dbeta:" + dbeta.getRows() + ":" + dbeta.getColumns());
		// System.out.println("xhat:" + xhat.getRows() + ":" + xhat.getColumns());
		/*
		 * Matrix dgammabTranspose =
		 * axonsContext.getMatrixFactory().createMatrix(xhatTranspose.getRows(),
		 * xhatTranspose.getColumns()); for (int i = 0; i < xhatTranspose.getColumns();
		 * i++) { dgammabTranspose.putColumn(i, dgammaTranspose); }
		 */

		// Matrix dbeta = doutTranspose.rowSums();
		// Matrix dbetaTranspose = dbeta.transpose();
		Matrix dbetaColumnVector = dout.rowSums();
		
		
		if (axonsContext.getRegularisationLambda() != 0) {

			//LOGGER.debug("Calculating total regularisation Gradients");

				try (InterrimMatrix biasesCopy = scaleAndShiftAxons.getShiftColumnVector().dup().asInterrimMatrix()) {

					dbetaColumnVector.asEditableMatrix()
						.addi(biasesCopy.asEditableMatrix().muli(axonsContext.getRegularisationLambda()));
			
			}
		}
		
		/*
		 * Matrix dbetabTranspose =
		 * axonsContext.getMatrixFactory().createMatrix(xhatTranspose.getRows(),
		 * xhatTranspose.getColumns()); for (int i = 0; i < xhatTranspose.getColumns();
		 * i++) { dbetabTranspose.putColumn(i, dbetaTranspose);
		 */

		int num = xhat.getColumns();

		try (InterrimMatrix istdColumnVector = axonsContext.getMatrixFactory().createOnes(varianceColumnVector.getRows(), 1).asEditableMatrix().divi(getStdDevColumnVector(varianceColumnVector)).asInterrimMatrix()) {

			Matrix gammaColumn = scaleAndShiftAxons.getScaleColumnVector();
			
			try (InterrimMatrix xhatMulDGamma = xhat.mulColumnVector(dgammaColumnVector).asInterrimMatrix()) {
				try (InterrimMatrix gammaMulIstdDiviNum = gammaColumn.mulColumnVector(istdColumnVector).asEditableMatrix().divi(num).asInterrimMatrix()) {
					Matrix dx = dout.mul(num).asEditableMatrix().subi(xhatMulDGamma)
							.subiColumnVector(dbetaColumnVector)
							.muliColumnVector(gammaMulIstdDiviNum);

					NeuronsActivation dxn = new NeuronsActivationImpl(dx, outerGradient.getOutput().getFeatureOrientation());
					
					//outerGradient.getOutput().close();
					
					return new DirectedComponentGradientImpl<>(outerGradient.getTotalTrainableAxonsGradients(),
							() -> new AxonsGradientImpl(scaleAndShiftAxons, dgammaColumnVector, dbetaColumnVector), dxn);
				}
			
			}
			
	
		}
		
		}
	}

	@Override
	public List<ChainableDirectedComponentActivation<NeuronsActivation>> decompose() {
		return Arrays.asList(this);
	}

	@Override
	public float getTotalRegularisationCost() {
		float totalRegularisationCost = 0f;
		if (axonsContext.getRegularisationLambda() != 0) {

			//LOGGER.info("Calculating total regularisation cost");

			if (scaleAndShiftAxons instanceof TrainableAxons) {
				try (InterrimMatrix weightsWithoutBiases = scaleAndShiftAxons.getDetachedConnectionWeights().asInterrimMatrix()) {
					try (InterrimMatrix biases = scaleAndShiftAxons.getDetachedLeftToRightBiases().asInterrimMatrix()) {
						float regularisationCostForWeights = weightsWithoutBiases.asEditableMatrix().muli(weightsWithoutBiases).sum();
						float regularisationCostForBiases = biases.asEditableMatrix().muli(biases).sum();
						totalRegularisationCost = totalRegularisationCost
								+ ((axonsContext.getRegularisationLambda()) * (regularisationCostForBiases + regularisationCostForWeights)) / 2f ;
					}
				}
			}
		}
		return totalRegularisationCost;
	}

	@Override
	public NeuronsActivation getOutput() {
		return scaleAndShiftAxonsActivation.getOutput();
	}

	@Override
	public DirectedAxonsComponent<?, ?> getAxonsComponent() {
		return batchNormAxons;
	}

	@Override
	public double getAverageRegularisationCost() {
		if (axonsContext.getRegularisationLambda()!= 1d) {
			throw new UnsupportedOperationException("Reguarlisation of batch norm synapses not yet supported");
		}
		return 0;
	}
}
