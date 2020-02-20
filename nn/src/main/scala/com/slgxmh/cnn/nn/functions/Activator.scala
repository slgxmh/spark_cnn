package com.slgxmh.cnn.nn.functions

import breeze.linalg.{DenseMatrix, inv}
import breeze.numerics.exp
import com.slgxmh.cnn.nn.data.Tensor

trait Activator {
  def output(input: DenseMatrix[Double]): DenseMatrix[Double]

  def derivative(activated: DenseMatrix[Double]): DenseMatrix[Double]

  def output(input: Tensor): Tensor

  def derivative(activated: Tensor): Tensor
}

object Activator {
  def get(t: ActivatorType): Activator = {
    // todo
    t match {
      case SIGMOID => new Activator {
        override def output(input: DenseMatrix[Double]): DenseMatrix[Double] = {
          if (input != null) {
            var denom: DenseMatrix[Double] = exp(input :+= (-1))
            denom = inv(denom :+= 1.0)
            DenseMatrix.ones(input.rows, input.cols) * denom
          }
          null
        }

        override def derivative(activated: DenseMatrix[Double]): DenseMatrix[Double] = {
          if (activated != null) {
            var ret = DenseMatrix.ones(activated.rows, activated.cols)
            ret = ret - activated
            ret = ret * activated
          }
          null
        }

        override def output(input: Tensor): Tensor = {
          if (input != null) {
            val length = input.data.length
            val ret = Tensor.zeros(input.dimShape)
            for (i <- 0 until length) {
              val data = ret.data(i)
              ret.data(i) = data + output(input.data(i))
            }
            return ret
          }
          null
        }

        override def derivative(activated: Tensor): Tensor = {
          if (activated != null) {
            val length = activated.data.length
            val ret = Tensor.zeros(activated.dimShape)
            for (i <- 0 until length) {
              val data = ret.data(i)
              ret.data(i) = data + derivative(activated.data(i))
            }
            return ret;
          }
          null
        }
      }
    }
  }

  trait ActivatorType

  case object SIGMOID extends ActivatorType

  case object RECTIFIED_LINEAR extends ActivatorType

  case object SOFTMAX extends ActivatorType

  case object TANH extends ActivatorType

  case object NONE extends ActivatorType

}
