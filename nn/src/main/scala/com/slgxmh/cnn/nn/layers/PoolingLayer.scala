package com.slgxmh.cnn.nn.layers

import com.slgxmh.cnn.nn.conf.LayerConf
import com.slgxmh.cnn.nn.data.{Tensor, Weight}
import org.slf4j.{Logger, LoggerFactory}

class PoolingLayer(val kernelRow: Int,
                   val kernelCol: Int,
                   val stride: Int,
                   val dimIn: Array[Int] = null) extends BaseLayers {
  override def createWeight(conf: LayerConf, input: Array[Int]): Weight =
    new Weight(Tensor.zeros(calcOutputShape), Tensor.zeros(Array(1)))

  override def calcOutputShape: Array[Int] = ???

  override def generateOutput(weight: Weight, input: Tensor): Tensor = ???

  override def activate(output: Tensor): Tensor = ???

  override def deriveDelta(activated: Tensor, error: Tensor): Tensor = ???

  override def gradient(input: Tensor, error: Tensor): Weight = ???

  override def calculateBackprop(weight: Weight, error: Tensor): Tensor = ???
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
