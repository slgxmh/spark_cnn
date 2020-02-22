package com.slgxmh.cnn.nn.layers

import com.slgxmh.cnn.nn.conf.LayerConf
import com.slgxmh.cnn.nn.data.{Tensor, Weight}

trait BaseLayers {
  val dimIn: Array[Int]

  /**
   * initialization
   *
   * @param conf
   * @param input
   * @return
   */
  def createWeight(conf: LayerConf, input: Array[Int]): Weight

  /**
   * {kernels, channels, }
   *
   * @return
   */
  def calcOutputShape: Array[Int]

  /**
   * feedForward
   *
   * @param weight
   * @param input
   * @return
   */
  def generateOutput(weight: Weight, input: Tensor): Tensor

  /**
   *
   * @param output
   * @return
   */
  def activate(output: Tensor): Tensor

  /**
   * backPropagation
   * compute delta = f'(output) * error
   *
   * @param activated
   * @param error
   * @return
   */
  def deriveDelta(activated: Tensor, error: Tensor): Tensor

  /**
   * compute dJ/dw = input * delta
   *
   * @param input
   * @param error
   * @return
   */
  def gradient(input: Tensor, error: Tensor): Weight

  /**
   * compute backprop delta = transpose(w) * error
   *
   * @param weight
   * @param error
   * @return
   */
  def calculateBackprop(weight: Weight, error: Tensor): Tensor
}
