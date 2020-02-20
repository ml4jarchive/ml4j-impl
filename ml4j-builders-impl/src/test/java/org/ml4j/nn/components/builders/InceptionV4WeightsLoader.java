/*
 * Copyright 2020 the original author or authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.ml4j.nn.components.builders;

import org.ml4j.nn.axons.BiasVector;
import org.ml4j.nn.axons.FeaturesVector;
import org.ml4j.nn.axons.WeightsMatrix;

/**
 * Interface for helper to load Inception V4 weights
 * 
 * @author Michael Lavelle
 *
 */
public interface InceptionV4WeightsLoader {
	
	WeightsMatrix getDenseLayerWeights(String name, int rows, int columns);
	
	BiasVector getDenseLayerBiases(String name, int rows, int columns);

	WeightsMatrix getConvolutionalLayerWeights(String name, int width, int height, int inputDepth, int outputDepth);
	
	WeightsMatrix getBatchNormLayerWeights(String name, int inputDepth);
	
	BiasVector getBatchNormLayerBiases(String name, int inputDepth);
	
	FeaturesVector getBatchNormLayerMean(String name, int inputDepth);

	FeaturesVector getBatchNormLayerVariance(String name, int inputDepth);

}
