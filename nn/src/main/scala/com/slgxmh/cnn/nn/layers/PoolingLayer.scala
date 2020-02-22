package com.slgxmh.cnn.nn.layers

import breeze.linalg.DenseMatrix
import com.slgxmh.cnn.nn.conf.LayerConf
import com.slgxmh.cnn.nn.data.{Tensor, Weight}
import org.slf4j.{Logger, LoggerFactory}

class PoolingLayer(val kernelRow: Int,
                   val kernelCol: Int,
                   val stride: Int,
                   val dimIn: Array[Int] = null) extends BaseLayers {
  override def createWeight(conf: LayerConf, input: Array[Int]): Weight =
    new Weight(Tensor.zeros(calcOutputShape), Tensor.zeros(Array(1)))

  override def generateOutput(weight: Weight, input: Tensor): Tensor = {
    val channels = dimIn(1)
    val rowKernels = calcOutputShape(2)
    val colKernels = calcOutputShape(3)

    val subMat: DenseMatrix[Double] = null
    val poolOut = Tensor.zeros(calcOutputShape)
    // todo
    for (ch <- 0 until channels) {
      for (c <- 0 until colKernels) {
        for (r <- 0 until rowKernels) {
          subMat = input.slice(0, ch).get(RangeUtils.interval(r * stride, r * stride + kernelRow),
            RangeUtils.interval(c * stride, c * stride + kernelCol))
          poolOut.slice(0, ch).put(r, c, subMat.max)
          weight.w.slice(0, ch).put(r, c, subMat.argmax)
        }
      }
    }
    poolOut
  }

  override def activate(output: Tensor): Tensor = output

  override def deriveDelta(activated: Tensor, error: Tensor): Tensor = error

  override def gradient(input: Tensor, error: Tensor): Weight = null

  override def calculateBackprop(weight: Weight, error: Tensor): Tensor = {
    val propDelta = Tensor.zeros(dimIn)
    val dimOut = calcOutputShape
    for (ch <- 0 until dimOut(1)) {
      for (or <- 0 until dimOut(2)) {
        for (oc <- 0 until dimOut(3)) {
          val subIdx = weight.w.slice(0, ch).get(or, oc).toInt
          propDelta.slice(0, ch).put(or * stride + subIdx % kernelRow, oc * stride + subIdx / kernelCol,
            propDelta.slice(0, ch).get(or * stride + subIdx % kernelRow,
              oc * stride + subIdx / kernelCol) + error.slice(0, ch).get(or, oc))
        }
      }
    }
    propDelta
  }

  override def calcOutputShape: Array[Int] =
    Array[Int](dimIn(0), dimIn(1), (dimIn(2) - kernelRow) / stride + 1, (dimIn(3) - kernelCol) / stride + 1)
}

object PoolingLayer {
  private val log: Logger = LoggerFactory.getLogger(this.getClass)

  def newPollingLayer(inputShape: Array[Int], conf: LayerConf): PoolingLayer = {
    val kernelRow = conf.get("kernel_row").asInstanceOf[Int]
    val kernelCol = conf.get("kernel_col").asInstanceOf[Int]
    val stride = conf.get("stride").asInstanceOf[Int]
    val p = new PoolingLayer(kernelRow, kernelCol, stride)
    log.info("create a polling layer")
    p
  }
}
