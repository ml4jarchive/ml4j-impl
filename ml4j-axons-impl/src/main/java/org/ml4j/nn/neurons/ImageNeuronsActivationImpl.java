package org.ml4j.nn.neurons;

import org.ml4j.FloatModifier;
import org.ml4j.FloatPredicate;
import org.ml4j.Matrix;
import org.ml4j.MatrixFactory;
import org.ml4j.images.Image;
import org.ml4j.images.MultiChannelImage;
import org.ml4j.images.SingleChannelImage;

public class ImageNeuronsActivationImpl extends NeuronsActivationImpl implements ImageNeuronsActivation {
	
	private Neurons3D neurons;
	private Image image;
	private boolean immutable;

	public ImageNeuronsActivationImpl(Matrix activations, Neurons3D neurons, NeuronsActivationFeatureOrientation featureOrientation, boolean immutable) {
		super(null, featureOrientation, immutable);
		this.neurons = neurons;
		if (neurons.getDepth() == 1) {
			image = new SingleChannelImage(activations.getRowByRowArray(), 
					 0, neurons.getHeight(), neurons.getWidth(), 0, 0, activations.getColumns());
		} else {
			 image = new MultiChannelImage(activations.getRowByRowArray(),
					 neurons.getDepth(), neurons.getHeight(), neurons.getWidth(), 0, 0, activations.getColumns());
		}
	}
	
	
	
	
	@Override
	public void setImmutable(boolean immutable) {
		this.immutable = immutable;
	}




	@Override
	public boolean isImmutable() {
		return immutable;
	}




	@Override
	public int getExampleCount() {
		return image.getExamples();
	}



	@Override
	 public ImageNeuronsActivation asImageNeuronsActivation(Neurons3D neurons) {
		  return this;
	  }
	
	@Override
	 public void close() {
		 image.close();
	  }
	
	
	
	@Override
	public Matrix getActivations(MatrixFactory matrixFactory) {
		/*
		if (immutable) {
			  throw new IllegalStateException("Immutable");
		  }
		  */
		Matrix activations =  matrixFactory.createMatrixFromRowsByRowsArray(getRows(), getColumns(), image.getData());
		activations.setImmutable(immutable);
		return activations;
	}
	
	@Override
	 public void applyValueModifier(FloatPredicate condition, FloatModifier modifier) {
		 	image.applyValueModifier(condition, modifier);
	  }
		
	@Override
	  public void applyValueModifier(FloatModifier modifier) {
		 	image.applyValueModifier(modifier);
	  }


	public ImageNeuronsActivationImpl(Neurons3D neurons, Image image, NeuronsActivationFeatureOrientation featureOrientation, boolean immutable) {
		super(null, featureOrientation, immutable);
		this.neurons = neurons;
		this.image = image;
	}
	
	
	
	@Override
	public NeuronsActivation dup() {
		return new ImageNeuronsActivationImpl(neurons, image, this.getFeatureOrientation(), immutable);
	}



	@Override
	public int getRows() {
		return neurons.getNeuronCountExcludingBias();
	}

	@Override
	public int getColumns() {
		return image.getExamples();
	}

	@Override
	public int getFeatureCount() {
		return getRows();
	}

	@Override
	public Neurons3D getNeurons() {
		return neurons;
	}
	
	public Image getImage() {
		return image;
	}
	
	
	  public Matrix im2Col(MatrixFactory matrixFactory, int filterHeight, int filterWidth, int strideHeight, int strideWidth, int paddingHeight, int paddingWidth) {
			Image imageWithPadding = image.softDup();
			imageWithPadding.setPaddingHeight(paddingHeight);
			imageWithPadding.setPaddingWidth(paddingWidth);
			return imageWithPadding.im2col(matrixFactory, filterHeight, filterWidth, strideHeight, strideWidth);
		}
	  
	  public Matrix im2Col2(MatrixFactory matrixFactory, int filterHeight, int filterWidth, int strideHeight, int strideWidth, int paddingHeight, int paddingWidth) {
			Image imageWithPadding = image.softDup();
			imageWithPadding.setPaddingHeight(paddingHeight);
			imageWithPadding.setPaddingWidth(paddingWidth);
			return imageWithPadding.im2col2(matrixFactory, filterHeight, filterWidth, strideHeight, strideWidth);
		}

	  @Override
	  public NeuronsActivation transpose() {
		  throw new UnsupportedOperationException();
	  }
}
