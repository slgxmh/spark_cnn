package com.slgxmh.cnn.nn.layers

import com.slgxmh.cnn.nn.conf.LayerConf
import com.slgxmh.cnn.nn.data.{Tensor, Weight}
import com.slgxmh.cnn.nn.functions.Activator
import com.slgxmh.cnn.nn.utils.ArrayUtils
import org.slf4j.{Logger, LoggerFactory}

/**
 * A convolutionLayer
 *
 * @param kernels
 * @param kernelRow
 * @param kernelCol
 * @param stride
 * @param padding
 * @param activator
 * @param dimIn
 */
class ConvolutionLayer(val kernels: Int,
                       val kernelRow: Int,
                       val kernelCol: Int,
                       val stride: Int,
                       val padding: Int,
                       val activator: Activator,
                       val dimIn: Array[Int] = null) extends BaseLayers {
  /**
   * todo
   *
   * @param conf
   * @param input dimension of input
   * @return
   */
  override def createWeight(conf: LayerConf, input: Array[Int]): Weight = {
    var typeW = conf.get("weight_type").asInstanceOf[Weight.WeightType]
    var typeB = conf.get("bias_type").asInstanceOf[Weight.WeightType]
    val dimIn = input(1) * input(2) * input(3)
    val valueW = typeW match {
      case Weight.XAVIER => Math.sqrt(2.0 / dimIn)
      case _ =>
        if (conf.get("weight_value") == null) Weight.DEFAULT_VALUE
        else conf.get("weight_value").asInstanceOf[Float];
    }
    val valueB = typeB match {
      case Weight.XAVIER => Math.sqrt(2.0 / dimIn)
      case _ =>
        if (conf.get("weight_value") == null) Weight.DEFAULT_VALUE
        else conf.get("weight_value").asInstanceOf[Float];
    }
    if (typeW == null) typeW = Weight.DEFAULT_TYPE
    if (typeB == null) typeB = Weight.DEFAULT_TYPE
    new Weight(Weight.create(typeW, valueW, Array(kernelRow * kernelCol * input(1), kernels)),
      Weight.create(typeB, valueB, Array(kernels)))
  }

  /**
   * todo
   *
   * @param weight
   * @param input
   * @return
   */
  override def generateOutput(weight: Weight, input: Tensor): Tensor = {
    val channels = dimIn(1)
    val rowKernels = calcOutputShape(2)
    val colKernels = calcOutputShape(3)
    val reshapeArr = Array[Double](kernelRow * kernelCol * channels * rowKernels * colKernels)
    val starPos = 0
    input = ArrayUtils.zeroPad(input, padding)
    /* reshaping to matrix to simplify convolution to normal matrix multiplication */
    for (i <- 0 until colKernels) {
      for (r <- 0 until rowKernels) {
        for (ch <- 0 until channels) {
          Array.copy(input)
        }
      }
    }
  }

  /**
   * 输出矩阵的结构
   *
   * @return
   */
  override def calcOutputShape: Array[Int] = {
    val dimOut = Array(
      dimIn(0),
      kernels,
      (dimIn(2) - kernelRow + 2 * padding) / stride + 1,
      (dimIn(3) - kernelRow + 2 * padding) / stride + 1
    )
    dimOut
  }

  override def activate(output: Tensor): Tensor = ???

  override def deriveDelta(activated: Tensor, error: Tensor): Tensor = ???

  override def gradient(input: Tensor, error: Tensor): Weight = ???

  override def calculateBackprop(weight: Weight, error: Tensor): Tensor = ???
}

object ConvolutionLayer {
  private val log: Logger = LoggerFactory.getLogger(this.getClass)

  def newConvolutionLayer(shape: Array[Int], conf: LayerConf): ConvolutionLayer = {
    val kernels = conf.get("num_output").asInstanceOf[Int]
    val kernelRow = conf.get("kernel_row").asInstanceOf[Int]
    val kernelCol = conf.get("kernel_col").asInstanceOf[Int]
    val stride = conf.get("stride").asInstanceOf[Int]
    val padding = conf.get("zeroPad").asInstanceOf[Int]
    val activator: Activator = Activator.get(conf.get("activator").asInstanceOf[Activator.ActivatorType])
    val c = new ConvolutionLayer(kernels, kernelRow, kernelCol, stride, padding, activator)
    log.info("create a convolutionLayer")
    c
  }
}
