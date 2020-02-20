package com.slgxmh.cnn.nn.utils

import breeze.linalg.DenseMatrix
import com.slgxmh.cnn.nn.data.Tensor

object ArrayUtils {
  /**
   *
   * @param in
   * @param padding
   * @return
   */
  def zeroPad(in: Tensor, padding: Int): Tensor = {
    val length = in.data.length
    val shape = in.dimShape
    val ret: Tensor = Tensor.zeros(Array(shape(0), shape(1), shape(2) + 2 * padding, shape(3) + 2 * padding))
    for (i <- 0 until length) ret.data(i) = zeroPadMatrix(in.data(i), padding)
    ret
  }

  /**
   *
   * @param in
   * @param padding
   * @return
   */
  private def zeroPadMatrix(in: DenseMatrix[Double], padding: Int): DenseMatrix[Double] = {
    val f = in
    val padded = Array[Double]((f.rows + 2 * padding) * (f.cols + 2 * padding))
    val org = f.toArray
    for (i <- 0 until f.cols) {
      val org_start_pos = i * f.rows
      val dest_start_pos = (i + padding) * (f.rows + 2 * padding) + padding
      Array.copy(org, org_start_pos, padded, dest_start_pos, f.rows)
    }
    new DenseMatrix[Double](f.rows + 2 * padding, f.cols + 2 * padding, padded)
  }
}
