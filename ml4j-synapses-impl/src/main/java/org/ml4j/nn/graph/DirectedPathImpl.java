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
 * Default implementation of DirectedPath
 * 
 * @author Michael Lavelle
 *
 * @param <E> The type of edge in this Path.
 */
public class DirectedPathImpl<E> implements DirectedPath<E> {

  private List<E> edges;

  public DirectedPathImpl(List<E> edges) {
    this.edges = edges;
  }

  public DirectedPathImpl(E edge) {
    this.edges = new ArrayList<>();
    this.edges.add(edge);
  }

  public DirectedPathImpl() {
    this(new ArrayList<>());
  }

  @Override
  public List<E> getEdges() {
    return edges;
  }

  @Override
  public void addEdge(E edge) {
    this.edges.add(edge);
  }
}
