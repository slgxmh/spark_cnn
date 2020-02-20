package com.slgxmh.cnn.nn.layers

import com.slgxmh.cnn.nn.conf.LayerConf
import com.slgxmh.cnn.nn.data.{Tensor, Weight}

trait BaseLayers {
  val dimIn: Array[Int]

  // initialization
  def createWeight(conf: LayerConf, input: Array[Int]): Weight

  def calcOutputShape: Array[Int]

  // feedForward
  def generateOutput(weight: Weight, input: Tensor): Tensor

  def activate(output: Tensor): Tensor

  // backPropagation
  def deriveDelta(activated: Tensor, error: Tensor): Tensor // compute delta = f'(output) * error

  def gradient(input: Tensor, error: Tensor): Weight // compute dJ/dw = input * delta

  def calculateBackprop(weight: Weight, error: Tensor): Tensor // compute backprop delta = transpose(w) * error
}
