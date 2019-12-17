/*
 * Copyright 2017 the original author or authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
 * in compliance with the License. You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software distributed under the License
 * is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
 * or implied. See the License for the specific language governing permissions and limitations under
 * the License.
 */

package org.ml4j.nn.neurons;

import java.io.ByteArrayOutputStream;
import java.io.PrintWriter;

import org.ml4j.EditableMatrix;
import org.ml4j.FloatModifier;
import org.ml4j.FloatPredicate;
import org.ml4j.InterrimMatrix;
import org.ml4j.Matrix;
import org.ml4j.MatrixFactory;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Encapsulates the activation activities of a set of Neurons.
 * 
 * @author Michael Lavelle
 */
public class NeuronsActivationImpl implements NeuronsActivation {

  private static final Logger LOGGER = LoggerFactory.getLogger(NeuronsActivationImpl.class);
  
  /**
   * The matrix of activations.
   */
  protected Matrix activations;
  private boolean immutable;
    
  private String stackTrace;
  
  /**
   * Defines whether the features of the activations are represented by the columns
   * or the rows of the activations Matrix.
   */
  private NeuronsActivationFeatureOrientation featureOrientation;
  
  
  public Neurons getNeurons() {
	  return null;
  }
  
  public void setImmutable(boolean immutable) {
	  this.immutable = immutable;
  }
  
  public boolean isImmutable() {
	  return immutable;
  }
  
  public void close() {
	  activations.close();
  }

  public Matrix getActivations(MatrixFactory matrixFactory) {
	  if (!activations.isImmutable()) {
		  activations.setImmutable(immutable);
	  }
	  return activations;
  }
  
  
  public void applyValueModifier(FloatPredicate condition, FloatModifier modifier) {
	  EditableMatrix editableActivations = activations.asEditableMatrix();
	  for (int i = 0; i < activations.getLength(); i++) {
			  if (condition.test(activations.get(i))) {
				  editableActivations.put(i, modifier.acceptAndModify(activations.get(i)));
			  }
	  }
  }
	
  public void applyValueModifier(FloatModifier modifier) {
	  EditableMatrix editableActivations = activations.asEditableMatrix();
	  for (int i = 0; i < activations.getLength(); i++) {
			  editableActivations.put(i, modifier.acceptAndModify(activations.get(i)));
	  }
  }
  
  public String getStackTrace() {
	  return stackTrace;
  }
  
  /**
   * Constructs a NeuronsActivation instance from a matrix of activations.
   * 
   * @param activations A matrix of activations
   * @param featureOrientation The orientation of the features of the activation matrix
   */
  public NeuronsActivationImpl(Matrix activations,
      NeuronsActivationFeatureOrientation featureOrientation) {
	  this(activations, featureOrientation, activations.isImmutable());
  }

  /**
   * Constructs a NeuronsActivation instance from a matrix of activations.
   * 
   * @param activations A matrix of activations
   * @param featureOrientation The orientation of the features of the activation matrix
   */
  public NeuronsActivationImpl(Matrix activations,
      NeuronsActivationFeatureOrientation featureOrientation, boolean immutable) {
    LOGGER.debug("Creating new NeuronsActivation");
    this.activations = activations;
    this.featureOrientation = featureOrientation;
	ByteArrayOutputStream os = new ByteArrayOutputStream(); 
	PrintWriter s = new
	PrintWriter(os); 
	new RuntimeException().printStackTrace(s); 
	s.flush();
	s.close(); 
	stackTrace = os.toString(); 

    this.immutable = immutable;
    if (activations != null && activations.isImmutable()) {
    	if (!immutable) {
    		throw new IllegalArgumentException();
    	}
    } else if (activations != null) {
    	activations.setImmutable(immutable);
    }
  }
  
  public void combineFeaturesInline(NeuronsActivation other) {
	  if (other.getFeatureOrientation() != featureOrientation) {
		  throw new IllegalArgumentException("Incompatible orientations");
	  }
	  if (featureOrientation == NeuronsActivationFeatureOrientation.ROWS_SPAN_FEATURE_SET) {
		  activations = activations.appendVertically(other.getActivations(null));
	  } else {
		  try (InterrimMatrix previousActivations = this.activations.asInterrimMatrix()) {
			  this.activations = previousActivations.appendHorizontally(other.getActivations(null));
		  }
	  }
  }
  
  
  public ImageNeuronsActivation asImageNeuronsActivation(Neurons3D neurons) {
	  return new ImageNeuronsActivationImpl(activations, neurons, featureOrientation, activations.isImmutable());
  }
  
  public void addInline(MatrixFactory matrixFactory, NeuronsActivation other) {
	  EditableMatrix editableActivations = activations.asEditableMatrix();
	  if (other.getFeatureOrientation() != featureOrientation) {
		  throw new IllegalArgumentException("Incompatible orientations");
	  }
	  if (other.getFeatureCount() != getFeatureCount()) {
		  throw new IllegalArgumentException("Incompatible activations");
	  }
	  if (other.getExampleCount() != getExampleCount()) {
		  throw new IllegalArgumentException("Incompatible activations");
	  }
	  editableActivations.addi(other.getActivations(matrixFactory));
  }

  /**
   * Obtain the feature orientation of the Matrix representing the activations - whether the
   * features are represented by the rows or the columns.
   * 
   * @return the feature orientation of the Matrix representing the activations - whether the
   *         features are represented by the rows or the columns
   */
  public NeuronsActivationFeatureOrientation getFeatureOrientation() {
    return featureOrientation;
  }
  
  public NeuronsActivation dup() {
	  return new NeuronsActivationImpl(activations.dup(), featureOrientation);
  }
  
  public int getRows() {
	  return activations.getRows();
  }
  
  public int getColumns() {
	  return activations.getColumns();
  }
  
  public NeuronsActivation filterActivationsByFeatureIndexRange(int startIndex, int endIndex) {
	  	int num = endIndex - startIndex;
		int[] indexes = new int[num];
		for (int j = 0; j < indexes.length; j++) {
			indexes[j] = j + startIndex;
		}
		return new NeuronsActivationImpl(featureOrientation == NeuronsActivationFeatureOrientation.ROWS_SPAN_FEATURE_SET ? activations.getRows(indexes) :
			activations.getColumns(indexes), featureOrientation);
  }
  /*
  public Matrix getActivations() {
    return activations;
  }
  */
  
  public NeuronsActivation transpose() {
	  if (featureOrientation == NeuronsActivationFeatureOrientation.COLUMNS_SPAN_FEATURE_SET) {
	      return new NeuronsActivationImpl(activations.transpose(), NeuronsActivationFeatureOrientation.ROWS_SPAN_FEATURE_SET);
	    } else {
		  return new NeuronsActivationImpl(activations.transpose(), NeuronsActivationFeatureOrientation.COLUMNS_SPAN_FEATURE_SET);
	    }
  }

  /**
   * Obtain the number of features ( including any bias ) represented by this NeuronsActivation.
   * 
   * @return the number of features ( including any bias ) represented by this NeuronsActivation.
   */
  public int getFeatureCount() {

    if (featureOrientation == NeuronsActivationFeatureOrientation.COLUMNS_SPAN_FEATURE_SET) {
      int featureCount = activations.getColumns();
      return featureCount;
    } else {
      int featureCount = activations.getRows();
      return featureCount;
    }
  }

@Override
public int getExampleCount() {
	return getColumns();
}
}
