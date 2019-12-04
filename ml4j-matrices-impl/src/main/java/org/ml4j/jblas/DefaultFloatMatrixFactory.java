package org.ml4j.jblas;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicLong;

import org.jblas.FloatMatrix;

public class DefaultFloatMatrixFactory implements FloatMatrixFactory {

	  /**
	 * Default serialization id
	 */
	private static final long serialVersionUID = 1L;

	private static AtomicLong matrixCount = new AtomicLong(0);
	  
	  public static Map<MatrixKey, List<FloatMatrix>> cache = new HashMap<>();
	 
	
	 private static synchronized void addMatrixToCache(FloatMatrix matrix) {
		  //matrix.subi(matrix);
		  //addMatrixToCacheSync(matrix);
		  matrixCount.decrementAndGet();
	  }
	  
	  private static synchronized void addMatrixToCacheSync(FloatMatrix matrix) {
		  List<FloatMatrix> matrices = cache.get(new MatrixKey(matrix.getRows(), matrix.getColumns()));
		  if (matrices == null) {
			  matrices = new ArrayList<>();
			  cache.put(new MatrixKey(matrix.getRows(), matrix.getColumns()), matrices);
		  }
		  if (matrices.size() < 3) {
		  matrices.add(matrix);
		  }
	  }
	  
	  public static synchronized FloatMatrix getMatrixFromCache(int rows, int cols) {
		  //System.out.println(matrixCount.get());
		  
		  List<FloatMatrix> matrices = cache.get(new MatrixKey(rows, cols));
		  if (matrices != null && !matrices.isEmpty()) {
			  int index = matrices.size() - 1;
			  FloatMatrix matrix =  matrices.remove(index);
			  matrix.subi(matrix);
			  return matrix;
		  }
		  //System.out.println("Unavailable from cache");
		  //new RuntimeException("blah").printStackTrace();
		  return null;
	  }
	  
	  
	  public static class MatrixKey {
		  
		  private int rows;
		  private int cols;
		  
		public MatrixKey(int rows, int cols) {
			super();
			this.rows = rows;
			this.cols = cols;
		}

		@Override
		public int hashCode() {
			final int prime = 31;
			int result = 1;
			result = prime * result + cols;
			result = prime * result + rows;
			return result;
		}

		@Override
		public boolean equals(Object obj) {
			if (this == obj)
				return true;
			if (obj == null)
				return false;
			if (getClass() != obj.getClass())
				return false;
			MatrixKey other = (MatrixKey) obj;
			if (cols != other.cols)
				return false;
			if (rows != other.rows)
				return false;
			return true;
		}
	  }


	@Override
	public FloatMatrix create(float[][] data) {
		return new FloatMatrix(data);
	}

	@Override
	public FloatMatrix create(int rows, int columns) {
		return new FloatMatrix(rows, columns);
	}

	@Override
	public FloatMatrix create(int rows, int columns, float[] data) {
		return new FloatMatrix(rows, columns, data);
	}
	
}
