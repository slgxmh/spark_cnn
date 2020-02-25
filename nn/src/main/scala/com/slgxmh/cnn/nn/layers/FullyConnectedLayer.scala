package com.slgxmh.cnn.nn.layers

import com.slgxmh.cnn.nn.conf.LayerConf
import com.slgxmh.cnn.nn.data.Weight.WeightType
import com.slgxmh.cnn.nn.data.{Tensor, Weight}
import com.slgxmh.cnn.nn.functions.Activator
import com.slgxmh.cnn.nn.functions.Activator.ActivatorType
import org.slf4j.{Logger, LoggerFactory}

class FullyConnectedLayer(val dimOut: Int,
                          val activator: Activator,
                          val dimIn: Array[Int] = null
                         ) extends BaseLayers {
  override def createWeight(conf: LayerConf, input: Array[Int]): Weight = {
    val dimIn = input(1) * input(2) * input(3)
    var typeW = conf.get("weight_type").asInstanceOf[WeightType];
    var typeB = conf.get("weight_type").asInstanceOf[WeightType];
    var valueW, ValueB: Double = Nil
    typeW match {
      case Weight.XAVIER => valueW = Math.sqrt(2.0 / dimIn)
      case _ =>
        valueW = if (conf.get("weight_value") == null) Weight.DEFAULT_VALUE
        else conf.get("weight_value").asInstanceOf[Float]
    }
    typeB match {
      case Weight.XAVIER => ValueB = Math.sqrt(2.0 / dimIn)
      case _ =>
        ValueB = if (conf.get("weight_value") == null) Weight.DEFAULT_VALUE
        else conf.get("weight_value").asInstanceOf[Float]
    }
    if (typeW == null) typeW = Weight.DEFAULT_TYPE
    if (typeB == null) typeB = Weight.DEFAULT_TYPE
    new Weight(Weight.create(typeW, valueW, Array(dimIn, dimOut)),
      Weight.create(typeW, valueW, Array(dimOut)))
  }

  override def calcOutputShape: Array[Int] = {
    Array[Int](dimIn(0), 1, 1, dimOut)
  }

  override def generateOutput(weight: Weight, input: Tensor): Tensor = {
    val data =
  }

  override def activate(output: Tensor): Tensor = ???

  override def deriveDelta(activated: Tensor, error: Tensor): Tensor = ???

  override def gradient(input: Tensor, error: Tensor): Weight = ???

  override def calculateBackprop(weight: Weight, error: Tensor): Tensor = ???
}

object FullyConnectedLayer {
  private val log: Logger = LoggerFactory.getLogger(this.getClass)

  def create(inputShape: Array[Int], conf: LayerConf): FullyConnectedLayer = {
    val dimOut = conf.get("num_output").asInstanceOf[Integer]
    val activator = Activator.get(conf.get("activator").asInstanceOf[ActivatorType])
    val f = new FullyConnectedLayer(dimOut, activator)
    log.info("Create a fully connected layer.")
    f
  }
}