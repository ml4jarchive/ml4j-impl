package org.ml4j.nn.axons;

import com.github.fommil.netlib.BLAS;

public class ThreadSafeNativeBlas {
	
	  public static synchronized void sgemm(String transa, String transb, int m, int n, int k, float alpha, float[] a, int aIdx, int lda, float[] b, int bIdx, int ldb, float beta, float[] c, int cIdx, int ldc) {
		  BLAS.getInstance().sgemm(transa, transb, m, n, k, alpha, a, aIdx, lda, b, bIdx, ldb, beta, c, cIdx, ldc);
	  }

}
