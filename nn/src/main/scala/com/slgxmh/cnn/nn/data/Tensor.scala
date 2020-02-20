package com.slgxmh.cnn.nn.data

import breeze.linalg.DenseMatrix

/**
 *
 * @param dimShape {kernels, channels, rows, cols}
 * @param length   kernels * channels
 * @param data     DenseMatrix[kernels * channels]
 */
class Tensor(var dimShape: Array[Int] = Array(1, 1, 1, 1),
             var length: Int = 0,
             var data: Array[DenseMatrix[Double]] = null) {
  def muli(d: Double): Tensor = {
    data.foreach(_ * d)
    this
  }

  /**
   * todo
   *
   * @param d
   * @return
   */
  def subi(d: Double): Tensor = {
    data.foreach(_ - d)
    this
  }

  /**
   * todo
   *
   * @param kernelIdx
   * @param channelIdx
   * @return
   */
  def slice(kernelIdx: Int, channelIdx: Int): DenseMatrix[Double] = {
    data(index(kernelIdx, channelIdx))
  }

  /**
   * todo
   *
   * @param kernelIdx
   * @param channelIdx
   * @return
   */
  private def index(kernelIdx: Int, channelIdx: Int): Int = {
    kernelIdx * dimShape(1) + channelIdx
  }
}

object Tensor {
  def zeros(shape: Array[Int]): Tensor = {
    newTensor(ZEROS, shape)
  }

  def ones(shape: Array[Int]): Tensor = {
    newTensor(ONES, shape)
  }

  def rand(shape: Array[Int]): Tensor = {
    newTensor(UNIFORM, shape)
  }

  private def newTensor(init: init, newDim: Array[Int]): Tensor = {
    val t = Tensor.newTensor(newDim)
    for (i <- 0 until t.length) {
      init match {
        case ZEROS => t.data(i) = DenseMatrix.zeros(t.dimShape(2), t.dimShape(3))
        case ONES => t.data(i) = DenseMatrix.ones(t.dimShape(2), t.dimShape(3))
        case UNIFORM => t.data(i) = DenseMatrix.rand(t.dimShape(2), t.dimShape(3))
      }
    }
    t
  }

  private def newTensor(newDim: Array[Int]): Tensor = {
    val t = new Tensor()
    if (newDim != null) {
      if (newDim.length > 4)
        throw new IllegalStateException(String.format(
          "Only support (n <= 4) dimensional tensor, current: %d", newDim.length))
      // dimShape = {kernels, channels, rows, cols}
      System.arraycopy(newDim, 0, t.dimShape, 4 - newDim.length, newDim.length)
      t.length = t.dimShape(0) * t.dimShape(1)
      t.data = new Array[DenseMatrix[Double]](t.length)
    }
    t
  }

  trait init

  case object ZEROS extends init

  case object ONES extends init

  case object UNIFORM extends init

}

