package org.ml4j.nn.axons;

import org.ml4j.Matrix;
import org.ml4j.MatrixFactory;
import org.ml4j.nn.neurons.Neurons3D;
import org.ml4j.nn.neurons.NeuronsActivation;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class ZeroPaddedConvolutionalAxonsImpl 
    implements TrainableAxons<Neurons3D, Neurons3D, ConvolutionalAxons>, ConvolutionalAxons {

  /**
   * Default serialization id.
   */
  private static final long serialVersionUID = 1L;
  
  private static final Logger LOGGER =
      LoggerFactory.getLogger(ZeroPaddedConvolutionalAxonsImpl.class);
  
  private UnpaddedConvolutionalAxonsImpl unpaddedConvolutionalAxonsImpl;
  private int zeroPadding;
  private Neurons3D leftNeurons;
  private Neurons3D rightNeurons;
  private int stride;


  /**
   * @param leftNeurons The left Neurons
   * @param rightNeurons The right Neurons
   * @param stride The stride
   * @param zeroPadding The amount of zero padding.
   * @param matrixFactory The MatrixFactory to use to initialise the weights.
   */
  public ZeroPaddedConvolutionalAxonsImpl(Neurons3D leftNeurons, Neurons3D rightNeurons, 
      int stride, int zeroPadding, 
      MatrixFactory matrixFactory) {
    this.leftNeurons = leftNeurons;
    this.rightNeurons = rightNeurons;
    this.zeroPadding = zeroPadding;
    this.stride = stride;
    validate();
    this.unpaddedConvolutionalAxonsImpl 
        = new UnpaddedConvolutionalAxonsImpl(new Neurons3D(
            leftNeurons.getWidth() + 2 * zeroPadding, leftNeurons.getHeight() + 2 * zeroPadding, 
            leftNeurons.getDepth(), leftNeurons.hasBiasUnit()), 
            rightNeurons, stride, matrixFactory);
  }

  /**
   * @param leftNeurons The left neurons.
   * @param rightNeurons The right neurons.
   * @param stride The stride.
   * @param zeroPadding The amount of zero padding.
   * @param matrixFactory The matrix factory.
   * @param connectionWeights The initial connection weights.
   */
  public ZeroPaddedConvolutionalAxonsImpl(Neurons3D leftNeurons, Neurons3D rightNeurons, 
      int stride, int zeroPadding, 
      MatrixFactory matrixFactory, Matrix connectionWeights) {
    this.leftNeurons = leftNeurons;
    this.rightNeurons = rightNeurons;
    this.zeroPadding = zeroPadding;
    this.stride = stride;
    validate();
    this.unpaddedConvolutionalAxonsImpl 
        = new UnpaddedConvolutionalAxonsImpl(new Neurons3D(
            leftNeurons.getWidth() + 2 * zeroPadding, leftNeurons.getHeight() + 2 * zeroPadding, 
            leftNeurons.getDepth(), leftNeurons.hasBiasUnit()), 
            rightNeurons, stride, matrixFactory, connectionWeights);
  }

  protected ZeroPaddedConvolutionalAxonsImpl(Neurons3D leftNeurons, Neurons3D rightNeurons,
       int stride, int zeroPadding, Matrix connectionWeights , 
       ConnectionWeightsMask connectionWeightsMask) {
    this.leftNeurons = leftNeurons;
    this.rightNeurons = rightNeurons;
    this.zeroPadding = zeroPadding;
    this.stride = stride;
    validate();
    this.unpaddedConvolutionalAxonsImpl 
        = new UnpaddedConvolutionalAxonsImpl(new Neurons3D(
            leftNeurons.getWidth() + 2 * zeroPadding, leftNeurons.getHeight() + 2 * zeroPadding, 
            leftNeurons.getDepth(), leftNeurons.hasBiasUnit()), 
        rightNeurons, stride, connectionWeights, connectionWeightsMask);
  }

  private void validate() {

    int inputWidth = leftNeurons.getWidth() + 2 * zeroPadding;
    int outputWidth = getRightNeurons().getWidth();

    int filterWidth = inputWidth + (1 - outputWidth) * (getStride());
    if (filterWidth <= 0) {
      throw new IllegalStateException(
          "Filter width calculated as:" + filterWidth + " but must be greater than 0");
    } else {
      LOGGER.debug("Filter width calculated as:" + filterWidth);
    }
    int validOutputWidth =
        (int) (((double) (inputWidth - filterWidth)) / ((double) getStride())) + 1;

    if (validOutputWidth != outputWidth) {
      throw new IllegalStateException("Invalid configuration");
    }

  }
  
  public int getStride() {
    return stride;
  }
  
  
  
  @Override
  public ConvolutionalAxons dup() {
    LOGGER.debug("Duplicting ConvolutionalAxons");
    return new ZeroPaddedConvolutionalAxonsImpl(leftNeurons, rightNeurons, stride, zeroPadding,
        unpaddedConvolutionalAxonsImpl.getDetachedConnectionWeights(),
        unpaddedConvolutionalAxonsImpl.connectionWeightsMask);
  }

  @Override
  public void adjustConnectionWeights(Matrix connectionWeights, 
      ConnectionWeightsAdjustmentDirection adjustmentDirection) {
    unpaddedConvolutionalAxonsImpl.adjustConnectionWeights(connectionWeights, adjustmentDirection);
  }

  @Override
  public Matrix getDetachedConnectionWeights() {
    return unpaddedConvolutionalAxonsImpl.getDetachedConnectionWeights();
  }

  @Override
  public Neurons3D getLeftNeurons() {
    return leftNeurons;
  }

  @Override
  public Neurons3D getRightNeurons() {
    return rightNeurons;
  }

  @Override
  public AxonsActivation pushLeftToRight(NeuronsActivation arg0, AxonsActivation arg1,
      AxonsContext arg2) {
   
    return unpaddedConvolutionalAxonsImpl
        .pushLeftToRight(pad(arg0, arg2), arg1, arg2);
  }

  private NeuronsActivation pad(NeuronsActivation input, AxonsContext context) {
    int paddedChannelFeatureCountWithoutBias = (getLeftNeurons().getWidth() + 2 * zeroPadding)
        * (getLeftNeurons().getHeight() + 2 * zeroPadding);

    int unpaddedChannelFeatureCountWithoutBias =
        (getLeftNeurons().getWidth()) * (getLeftNeurons().getHeight());
    
    int paddingTotalFetureCountWithoutBias =
        paddedChannelFeatureCountWithoutBias * getLeftNeurons().getDepth();
    Matrix paddedWithoutBias = context.getMatrixFactory()
        .createZeros(input.getActivations().getRows(), 
            paddingTotalFetureCountWithoutBias);
    for (int inputChannel = 0; inputChannel < getLeftNeurons().getDepth(); inputChannel++) {
      for (int w = 0; w < getLeftNeurons().getWidth() + 2 * zeroPadding; w++) {
        for (int h = 0; h < getLeftNeurons().getHeight() + 2 * zeroPadding; h++) {
          if (w >= zeroPadding && w < (getLeftNeurons().getWidth() + zeroPadding)
              && h >= zeroPadding && h < (getLeftNeurons().getHeight() + zeroPadding)) {

            int paddedWithoutBiasFeatureIndex = inputChannel * paddedChannelFeatureCountWithoutBias
                + (h * (getLeftNeurons().getWidth() + 2 * zeroPadding)) + w;

            int unpaddedWithoutBiasFeatureIndex = 
                unpaddedChannelFeatureCountWithoutBias * inputChannel
                + (h - zeroPadding) * getLeftNeurons().getWidth() + (w - zeroPadding);

            paddedWithoutBias.putColumn(paddedWithoutBiasFeatureIndex,
                input.getActivations().getColumn(unpaddedWithoutBiasFeatureIndex));
          }
        }
      }
    }

    LOGGER.debug("Padded from " + input.getActivations().getColumns() 
        + " to " + (paddedWithoutBias.getColumns()));

    return new NeuronsActivation(
        paddedWithoutBias,
        input.getFeatureOrientation());
  }
  
  private NeuronsActivation unpad(NeuronsActivation input, AxonsContext context) {
    
  
    int paddedChannelFeatureCountWithoutBias = (getLeftNeurons().getWidth() + 2 * zeroPadding)
        * (getLeftNeurons().getHeight() + 2 * zeroPadding);

    int unpaddedChannelFeatureCountWithoutBias =
        (getLeftNeurons().getWidth()) * (getLeftNeurons().getHeight());

    Matrix unpaddedWithoutBias = context.getMatrixFactory()
        .createZeros(unpaddedChannelFeatureCountWithoutBias, 
            input.getActivations().getColumns());
    for (int inputChannel = 0; inputChannel < getLeftNeurons().getDepth(); inputChannel++) {
      for (int w = 0; w < getLeftNeurons().getWidth() + 2 * zeroPadding; w++) {
        for (int h = 0; h < getLeftNeurons().getHeight() + 2 * zeroPadding; h++) {
          if (w >= zeroPadding && w < (getLeftNeurons().getWidth() + zeroPadding)
              && h >= zeroPadding && h < (getLeftNeurons().getHeight() + zeroPadding)) {

            int paddedWithoutBiasFeatureIndex =  
                 inputChannel * paddedChannelFeatureCountWithoutBias
                + (h * (getLeftNeurons().getWidth() + 2 * zeroPadding)) + w;

            int unpaddedWithoutBiasFeatureIndex = 
                + unpaddedChannelFeatureCountWithoutBias * inputChannel
                + (h - zeroPadding) * getLeftNeurons().getWidth() + (w - zeroPadding);

            //LOGGER.debug(inputChannel + ":" + w + ":" + h);
            //LOGGER.debug(unpaddedWithoutBiasFeatureIndex + ":" + paddedWithoutBiasFeatureIndex);

            unpaddedWithoutBias.putRow(unpaddedWithoutBiasFeatureIndex,
                input.getActivations().getRow(paddedWithoutBiasFeatureIndex));
          }
        }
      }
    }

    LOGGER.debug("Unpadded from " + input.getActivations().getColumns() + " to "
        + (unpaddedWithoutBias.getColumns()));

    return new NeuronsActivation(
        unpaddedWithoutBias,
        input.getFeatureOrientation());
  }

  @Override
  public AxonsActivation pushRightToLeft(NeuronsActivation arg0, AxonsActivation arg1,
      AxonsContext arg2) {
    AxonsActivation activated =  unpaddedConvolutionalAxonsImpl.pushRightToLeft(arg0, arg1, arg2);
    Matrix inputDropoutMask = activated.getInputDropoutMask();
    NeuronsActivation output = activated.getOutput();
    return new AxonsActivationImpl(this, inputDropoutMask, 
        activated.getPostDropoutInputWithPossibleBias(), unpad(output, arg2));
  }

  @Override
  public int getZeroPadding() {
    return zeroPadding;
  }
  
  @Override
  public boolean isTrainable(AxonsContext context) {
    return unpaddedConvolutionalAxonsImpl.isTrainable(context);
  }
}
