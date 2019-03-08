package org.apind.detectsprites

import org.apache.spark.rdd._
import org.opencv.core.{Core, Mat, MatOfRect, Point, Rect, Scalar}
import org.opencv.imgcodecs.Imgcodecs
import java.io._

import org.opencv.imgproc.Imgproc


object TemplateBasedSpriteDetector {
  def detect(rdd: RDD[(Int, String)]): RDD[String] = {
    // Hard code for now, dynamically generate later when I have more sprites
    val sprites: List[Mat] = List(
      Imgcodecs.imread(getClass().getResource("/sprites/SuperMarioBros-Nes/big_mario_0.png").getPath()),
      Imgcodecs.imread(getClass().getResource("/sprites/SuperMarioBros-Nes/big_mario_10.png").getPath()),
      Imgcodecs.imread(getClass().getResource("/sprites/SuperMarioBros-Nes/big_mario_11.png").getPath()),
      Imgcodecs.imread(getClass().getResource("/sprites/SuperMarioBros-Nes/big_mario_12.png").getPath()),
      Imgcodecs.imread(getClass().getResource("/sprites/SuperMarioBros-Nes/big_mario_13.png").getPath()),
      Imgcodecs.imread(getClass().getResource("/sprites/SuperMarioBros-Nes/big_mario_14.png").getPath()),
      Imgcodecs.imread(getClass().getResource("/sprites/SuperMarioBros-Nes/big_mario_15.png").getPath()),
      Imgcodecs.imread(getClass().getResource("/sprites/SuperMarioBros-Nes/big_mario_16.png").getPath()),
      Imgcodecs.imread(getClass().getResource("/sprites/SuperMarioBros-Nes/big_mario_17.png").getPath()),
      Imgcodecs.imread(getClass().getResource("/sprites/SuperMarioBros-Nes/big_mario_18.png").getPath()),
      Imgcodecs.imread(getClass().getResource("/sprites/SuperMarioBros-Nes/big_mario_19.png").getPath()),
      Imgcodecs.imread(getClass().getResource("/sprites/SuperMarioBros-Nes/big_mario_1.png").getPath()),
      Imgcodecs.imread(getClass().getResource("/sprites/SuperMarioBros-Nes/big_mario_20.png").getPath()),
      Imgcodecs.imread(getClass().getResource("/sprites/SuperMarioBros-Nes/big_mario_2.png").getPath()),
      Imgcodecs.imread(getClass().getResource("/sprites/SuperMarioBros-Nes/big_mario_3.png").getPath()),
      Imgcodecs.imread(getClass().getResource("/sprites/SuperMarioBros-Nes/big_mario_4.png").getPath()),
      Imgcodecs.imread(getClass().getResource("/sprites/SuperMarioBros-Nes/big_mario_5.png").getPath()),
      Imgcodecs.imread(getClass().getResource("/sprites/SuperMarioBros-Nes/big_mario_6.png").getPath()),
      Imgcodecs.imread(getClass().getResource("/sprites/SuperMarioBros-Nes/big_mario_7.png").getPath()),
      Imgcodecs.imread(getClass().getResource("/sprites/SuperMarioBros-Nes/big_mario_8.png").getPath()),
      Imgcodecs.imread(getClass().getResource("/sprites/SuperMarioBros-Nes/big_mario_9.png").getPath()),
      Imgcodecs.imread(getClass().getResource("/sprites/SuperMarioBros-Nes/lil_mario_0.png").getPath()),
      Imgcodecs.imread(getClass().getResource("/sprites/SuperMarioBros-Nes/lil_mario_10.png").getPath()),
      Imgcodecs.imread(getClass().getResource("/sprites/SuperMarioBros-Nes/lil_mario_11.png").getPath()),
      Imgcodecs.imread(getClass().getResource("/sprites/SuperMarioBros-Nes/lil_mario_12.png").getPath()),
      Imgcodecs.imread(getClass().getResource("/sprites/SuperMarioBros-Nes/lil_mario_13.png").getPath()),
      Imgcodecs.imread(getClass().getResource("/sprites/SuperMarioBros-Nes/lil_mario_1.png").getPath()),
      Imgcodecs.imread(getClass().getResource("/sprites/SuperMarioBros-Nes/lil_mario_2.png").getPath()),
      Imgcodecs.imread(getClass().getResource("/sprites/SuperMarioBros-Nes/lil_mario_3.png").getPath()),
      Imgcodecs.imread(getClass().getResource("/sprites/SuperMarioBros-Nes/lil_mario_4.png").getPath()),
      Imgcodecs.imread(getClass().getResource("/sprites/SuperMarioBros-Nes/lil_mario_5.png").getPath()),
      Imgcodecs.imread(getClass().getResource("/sprites/SuperMarioBros-Nes/lil_mario_6.png").getPath()),
      Imgcodecs.imread(getClass().getResource("/sprites/SuperMarioBros-Nes/lil_mario_7.png").getPath()),
      Imgcodecs.imread(getClass().getResource("/sprites/SuperMarioBros-Nes/lil_mario_8.png").getPath()),
      Imgcodecs.imread(getClass().getResource("/sprites/SuperMarioBros-Nes/lil_mario_9.png").getPath())
    )
    // This task is not serializable...
    rdd.map({
      case (i: Int, src: String) => {
        val image = Imgcodecs.imread(src)
        val s = sprites
        // Generate a list of heatmap like images denoting where the corresponding template (sprite) is in the source image.
        // This block causes serialization to fail
        val tmplLocMat = sprites.map(sprite => {
          var out = new Mat()
          Imgproc.matchTemplate(image, sprite, out, Imgproc.TM_CCOEFF)
          out
          new Mat()
        })

        // For each "heatmap" get the location of the maximum value, which corresponds to the sprite's most probably location
        val maxLocs = tmplLocMat map(tmplHeatMap => {
          Core.minMaxLoc(tmplHeatMap)
        })
        // We know mario is only going to be in the image once, so we can pop off the single maximum value, and return
        // generated bounding box
        val spriteMaxLoc: Tuple2[Core.MinMaxLocResult, Mat] = maxLocs.zip(sprites).maxBy {
          case (mmlr: Core.MinMaxLocResult, sprite: Mat) => {
            mmlr.maxVal
          }
        }

        Imgproc.rectangle(image, spriteMaxLoc._1.maxLoc, new Point(spriteMaxLoc._1.maxLoc.x + spriteMaxLoc._2.cols(),
          spriteMaxLoc._1.maxLoc.y + spriteMaxLoc._2.rows()), new Scalar(255, 255, 255))
        val filename = s"/home/mcsmash/dev/deep_thought/data_tools/preprocessing/detect-sprites/output/${i}.png"
        Imgcodecs.imwrite(filename, image)

        filename
      }
    })
  }
}
