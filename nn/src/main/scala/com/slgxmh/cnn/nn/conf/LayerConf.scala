package com.slgxmh.cnn.nn.conf

import com.slgxmh.cnn.nn.conf.LayerConf.LayerType

object LayerConf {

  trait LayerType

  case object CONVOLUTION extends LayerType

  case object POOLING extends LayerType

  case object FULLYCONN extends LayerType

}

class LayerConf(var layerParams: Map[String, Any],
                val layerType: LayerType) {
  def get(key: String): Any = {
    layerParams.get(key)
  }

  def set(key: String, value: Any): Unit = {
    layerParams += (key -> value)
  }
}

