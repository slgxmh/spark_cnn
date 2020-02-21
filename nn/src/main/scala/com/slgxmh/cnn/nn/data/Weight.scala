package com.slgxmh.cnn.nn.data

object Weight {
  val DEFAULT_VALUE: Double = 0.0
  val DEFAULT_TYPE: WeightType = CONSTANT

  def create(t: WeightType, value: Double, dimW: Array[Int]): Tensor = {
    var tensor: Tensor = null
    t match {
      case CONSTANT => tensor = Tensor.ones(dimW).muli(value)
      case UNIFORM => tensor = Tensor.rand(dimW).subi(0.5f).muli(2.0f * value)
    }
    tensor
  }

  trait WeightType

  case object CONSTANT extends WeightType

  case object UNIFORM extends WeightType

  case object GAUSSIAN extends WeightType

  case object XAVIER extends WeightType

}

/**
 *
 * @param w todo
 * @param b todo
 */
class Weight(val w: Tensor, val b: Tensor)
