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

package org.ml4j.nn.graph;

import java.util.ArrayList;
import java.util.List;

/**
 * Default implementation of DirectedDipoleGraph
 * 
 * @author Michael Lavelle
 *
 * @param <E> The type of edge in this graph.
 */
public class DirectedDipoleGraphImpl<E> implements DirectedDipoleGraph<E> {

  private List<DirectedPath<E>> parallelPaths;

  public DirectedDipoleGraphImpl(List<DirectedPath<E>> parallelPaths) {
    this.parallelPaths = parallelPaths;
  }
  
  public DirectedDipoleGraphImpl(E singleEdge) {
    this.parallelPaths = new ArrayList<>();
    this.parallelPaths.add(new DirectedPathImpl<E>(singleEdge));
  }

  public DirectedDipoleGraphImpl() {
    this(new ArrayList<>());
  }

  @Override
  public List<DirectedPath<E>> getParallelPaths() {
    return parallelPaths;
  }

  @Override
  public void addParallelPath(DirectedPath<E> parallelPath) {
    this.parallelPaths.add(parallelPath);
  }
}
