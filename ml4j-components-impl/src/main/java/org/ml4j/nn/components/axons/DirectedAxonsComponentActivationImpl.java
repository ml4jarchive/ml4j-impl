package org.ml4j.nn.components.axons;

import java.util.Arrays;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.function.Supplier;

import org.ml4j.EditableMatrix;
import org.ml4j.InterrimMatrix;
import org.ml4j.Matrix;
import org.ml4j.nn.axons.Axons;
import org.ml4j.nn.axons.AxonsActivation;
import org.ml4j.nn.axons.AxonsContext;
import org.ml4j.nn.axons.AxonsGradient;
import org.ml4j.nn.axons.AxonsGradientImpl;
import org.ml4j.nn.axons.TrainableAxons;
import org.ml4j.nn.components.ChainableDirectedComponentActivation;
import org.ml4j.nn.components.DirectedComponentGradient;
import org.ml4j.nn.components.DirectedComponentGradientImpl;
import org.ml4j.nn.neurons.NeuronsActivation;
import org.ml4j.nn.neurons.NeuronsActivationFeatureOrientation;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class DirectedAxonsComponentActivationImpl implements DirectedAxonsComponentActivation {

	private static final Logger LOGGER = LoggerFactory.getLogger(DirectedAxonsComponentActivationImpl.class);
	private static final ExecutorService executorService = Executors.newFixedThreadPool(2);

	private Axons<?, ?, ?> axons;
	private AxonsActivation axonsActivation;
	private AxonsContext axonsContext;
	private DirectedAxonsComponent<?, ?> directedAxonsComponent;

	public DirectedAxonsComponentActivationImpl(DirectedAxonsComponent<?, ?> directedAxonsComponent,
			AxonsActivation axonsActivation, AxonsContext axonsContext) {
		this.axonsActivation = axonsActivation;
		this.axons = axonsActivation.getAxons();
		this.axonsContext = axonsContext;
		this.directedAxonsComponent = directedAxonsComponent;
	}
	
	public AxonsActivation getAxonsActivation() {
		return axonsActivation;
	}

	@Override
	public DirectedComponentGradient<NeuronsActivation> backPropagate(
			DirectedComponentGradient<NeuronsActivation> outerGradient) {
		return backPropagateThroughAxons(outerGradient, axonsContext);
	}

	@Override
	public NeuronsActivation getOutput() {
		return axonsActivation.getOutput();
	}

	private DirectedComponentGradient<NeuronsActivation> backPropagateThroughAxons(
			DirectedComponentGradient<NeuronsActivation> dz, AxonsContext axonsContext) {

		LOGGER.debug("Pushing data right to left through axons...");


		if (dz.getOutput().getFeatureCount() != axons.getRightNeurons().getNeuronCountIncludingBias()) {
			throw new IllegalArgumentException("Incorrect feature count");
		}
		
		if (dz.getOutput().getFeatureOrientation() != NeuronsActivationFeatureOrientation.ROWS_SPAN_FEATURE_SET) {
			throw new IllegalArgumentException("Only ROWS_SPAN_FEATURE_SET supported");
		}

		// Will contain bias unit if Axons have left bias unit
		AxonsActivation inputGradientActivation = axons.pushRightToLeft(dz.getOutput(), axonsActivation, axonsContext);
		
		if (inputGradientActivation.getPostDropoutInput()
				.getFeatureOrientation() != NeuronsActivationFeatureOrientation.ROWS_SPAN_FEATURE_SET) {
			throw new IllegalArgumentException("Only ROWS_SPAN_FEATURE_SET supported");
		}
		if (axonsActivation.getPostDropoutInput()
				.getFeatureOrientation() != NeuronsActivationFeatureOrientation.ROWS_SPAN_FEATURE_SET) {
			throw new IllegalArgumentException("Only ROWS_SPAN_FEATURE_SET supported");
		}

		NeuronsActivation inputGradient = inputGradientActivation.getOutput();
		final Callable<AxonsGradient> grad;
		if (false) {//axons instanceof ConvolutionalAxonsLatest) {
			// grad = () -> createTotalTrainableAxonsGradient100(inputGradientActivation,
			//		axonsContext);
		} else {
			grad = () -> createTotalTrainableAxonsGradient(inputGradientActivation,
					axonsContext);
		}
		
		//Callable<AxonsGradient> grad = () -> createTotalTrainableAxonsGradient(inputGradientActivation,
		//		axonsContext);
		
		
		//Callable<AxonsGradient> grad = () -> null;
		
		Future<AxonsGradient> gradFuture = executorService.submit(grad);
                                               Supplier<AxonsGradient> totalTrainableAxonsGradient = () -> getFromFuture(gradFuture);
		/*
		Supplier<AxonsGradient> totalTrainableAxonsGradient = () -> {
			try {
				return grad.call();
			} catch (Exception e) {
				// TODO Auto-generated catch block
				throw new RuntimeException(e);
			}
		};
		*/

		//Supplier<AxonsGradient> totalTrainableAxonsGradient = gradFuture == null ? null : () -> getFromFuture(gradFuture);

		if (totalTrainableAxonsGradient != null) {
			return new DirectedComponentGradientImpl<>(dz.getTotalTrainableAxonsGradients(),
					totalTrainableAxonsGradient, inputGradient);
		} else {
			return new DirectedComponentGradientImpl<>(dz.getTotalTrainableAxonsGradients(), inputGradient);
		}
	}
	
	private AxonsGradient getFromFuture(Future<AxonsGradient> future) {
		try {
			return  future.get();
		} catch (InterruptedException e) {
			 throw new RuntimeException(e);
		} catch (ExecutionException e) {
			e.printStackTrace();
			 throw new RuntimeException(e);
		}
	}

	protected AxonsGradient createTotalTrainableAxonsGradient(AxonsActivation inputGradientActivation, AxonsContext axonsContext) {
		AxonsGradient totalTrainableAxonsGradient = null;
		EditableMatrix totalTrainableAxonsGradientMatrixNonBias = null;
		Matrix totalTrainableAxonsGradientMatrixBias = null;

		if (axons instanceof TrainableAxons<?, ?, ?> && axons.isTrainable(axonsContext)) {

			TrainableAxons<?, ?, ?> trainableAxons = (TrainableAxons<?, ?, ?>) axons;

			LOGGER.debug("Calculating Axons Gradients");
			
			Matrix first = inputGradientActivation.getPostDropoutInput().getActivations(axonsContext.getMatrixFactory());

			try (InterrimMatrix second = axonsActivation.getPostDropoutInput().getActivations(axonsContext.getMatrixFactory()).transpose().asInterrimMatrix()) {
				//System.out.println(first.getRows() + ":" + first.getColumns());
				//System.out.println(second.getRows() + ":" + second.getColumns());
				//System.out.println(first.getClass() + ":" + second.getClass());
				totalTrainableAxonsGradientMatrixNonBias = first.mmul(second).asEditableMatrix();
				//System.out.println("Done");
			}

			if (axons.getLeftNeurons().hasBiasUnit()) {
				totalTrainableAxonsGradientMatrixBias = first.rowSums();
			}

			if (axonsContext.getRegularisationLambda() != 0) {

				LOGGER.debug("Calculating total regularisation Gradients");

				try (InterrimMatrix connectionWeightsCopy = trainableAxons.getDetachedConnectionWeights().asInterrimMatrix()) {
					
					Matrix regularisationAddition = connectionWeightsCopy.asEditableMatrix().muli(axonsContext.getRegularisationLambda());
					
					totalTrainableAxonsGradientMatrixNonBias
							.addi(regularisationAddition);
						
				}
			}
			totalTrainableAxonsGradient = new AxonsGradientImpl((TrainableAxons<?, ?, ?>) axons,
					totalTrainableAxonsGradientMatrixNonBias, totalTrainableAxonsGradientMatrixBias);
		}
		return totalTrainableAxonsGradient;
	}
	
	/*
	protected AxonsGradient createTotalTrainableAxonsGradient100(AxonsActivation inputGradientActivation, AxonsContext axonsContext) {
		AxonsGradient totalTrainableAxonsGradient = null;
		EditableMatrix totalTrainableAxonsGradientMatrixNonBias = null;
		Matrix totalTrainableAxonsGradientMatrixBias = null;

		if (axons instanceof TrainableAxons<?, ?, ?> && axons.isTrainable(axonsContext)) {

			TrainableAxons<?, ?, ?> trainableAxons = (TrainableAxons<?, ?, ?>) axons;

			LOGGER.debug("Calculating Axons Gradients");
			
			ConvolutionalAxonsLatest latest = (ConvolutionalAxonsLatest)axons;
			Neurons3D left = (Neurons3D)axons.getLeftNeurons();
			Neurons3D right = (Neurons3D)axons.getRightNeurons();
			
			int inputWidthWithPadding = latest.getLeftNeurons().getWidth() + latest.config.getPaddingWidth() * 2;
			int inputHeightWithPadding = latest.getLeftNeurons().getHeight()  + latest.config.getPaddingHeight() * 2;
			int kernelWidth = inputWidthWithPadding + (1 - latest.getRightNeurons().getWidth()) * (latest.config.getStrideWidth());
			int kernelHeight = inputHeightWithPadding + (1 - latest.getRightNeurons().getHeight()) * (latest.config.getStrideHeight());

			NeuronsActivation firstAct = inputGradientActivation.getPostDropoutInput();
			EditableMatrix first = firstAct.getActivations().asEditableMatrix();
			int origRows = first.getRows();
			int origColumns = first.getColumns();
			try (InterrimMatrix second = axonsActivation.getPostDropoutInput().asImageNeuronsActivation(left).im2Col(axonsContext.getMatrixFactory(), kernelHeight, kernelWidth, latest.getStrideHeight(), 
					latest.getStrideWidth(), latest.config.getPaddingHeight(), latest.config.getPaddingWidth()).asInterrimMatrix()) {
			try (InterrimMatrix secondTranspose = second.transpose().asInterrimMatrix()) {
			first.reshape(right.getDepth(), first.getLength() /right.getDepth());

			totalTrainableAxonsGradientMatrixNonBias = first.mmul(secondTranspose).asEditableMatrix();
			first.reshape(origRows, origColumns);
			}
			}
			
			if (axons.getLeftNeurons().hasBiasUnit()) {
				totalTrainableAxonsGradientMatrixBias = first.rowSums();
			}

			if (axonsContext.getRegularisationLambda() != 0) {

				LOGGER.debug("Calculating total regularisation Gradients");

				try (InterrimMatrix connectionWeightsCopy = trainableAxons.getDetachedConnectionWeights().asInterrimMatrix()) {
					
					EditableMatrix regularisationAddition = connectionWeightsCopy.asEditableMatrix().muli(axonsContext.getRegularisationLambda()).asEditableMatrix();
					
					totalTrainableAxonsGradientMatrixNonBias
							.addi(regularisationAddition);
						
				}
			}
			totalTrainableAxonsGradient = new AxonsGradientImpl((TrainableAxons<?, ?, ?>) axons,
					totalTrainableAxonsGradientMatrixNonBias, totalTrainableAxonsGradientMatrixBias);
		}
		return totalTrainableAxonsGradient;
	}
	*/

	@Override
	public DirectedAxonsComponent<?, ?> getAxonsComponent() {
		return directedAxonsComponent;
	}

	@Override
	public float getTotalRegularisationCost() {

		float totalRegularisationCost = 0f;
		if (axonsContext.getRegularisationLambda() != 0) {

			LOGGER.info("Calculating total regularisation cost");

			if (axons instanceof TrainableAxons) {

				try (InterrimMatrix weightsWithoutBiases = ((TrainableAxons<?, ?, ?>) axons).getDetachedConnectionWeights().asInterrimMatrix()) {
					float regularisationMatrix = weightsWithoutBiases.asEditableMatrix().muli(weightsWithoutBiases).sum();
					totalRegularisationCost = totalRegularisationCost
							+ ((axonsContext.getRegularisationLambda()) * regularisationMatrix) / 2;
				}
			}
		}
		return totalRegularisationCost;
	}

	@Override
	public double getAverageRegularisationCost() {
		return getTotalRegularisationCost() / axonsActivation.getOutput().getActivations(axonsContext.getMatrixFactory()).getColumns();
	}

	@Override
	public List<ChainableDirectedComponentActivation<NeuronsActivation>> decompose() {
		return Arrays.asList(this);
	}
}
