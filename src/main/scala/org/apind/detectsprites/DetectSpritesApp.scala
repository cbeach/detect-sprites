package org.apind.detectsprites

import org.apache.spark.{SparkConf, SparkContext}
import org.opencv.core.{Core, Mat, MatOfRect, Point, Rect, Scalar}
import org.opencv.imgcodecs.Imgcodecs
import java.awt.Image
import java.awt.image.DataBufferByte


/**
  * Use this to test the app locally, from sbt:
  * sbt "run inputFile.txt outputFile.txt"
  *  (+ select DetectSpritesLocalApp when prompted)
  */
object DetectSpritesLocalApp extends App {
  val conf = new SparkConf()
    .setMaster("local")
    .setAppName("my awesome app")

  Runner.run(conf)
}

/**
  * Use this when submitting the app to a cluster with spark-submit
  * */

/*
object DetectSpritesApp extends App {
  // spark-submit command should supply all necessary config elements
  Runner.run(new SparkConf())
}
*/

object Runner {
  def run(conf: SparkConf): Unit = {
    System.loadLibrary(Core.NATIVE_LIBRARY_NAME)
    val sc = new SparkContext(conf)
    val frames: List[(Int, String)] = (1 until 99).map(frame_number => {
      (frame_number, getClass().getResource(s"/test-examples/SuperMarioBros-Nes/${frame_number}.png").getPath())
    }).toList

    val rdd = sc.makeRDD(frames)
    TemplateBasedSpriteDetector.detect(rdd)
  }
}
