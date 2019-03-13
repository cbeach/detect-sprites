package org.apind.detectsprites

import org.apache.spark.rdd._
import org.opencv.core.{Core, Mat, MatOfRect, Point, Rect, Scalar}
import org.opencv.imgcodecs.Imgcodecs
import java.io._

import org.apind.util.OpenCVMat
import org.opencv.imgproc.Imgproc


object TemplateBasedSpriteDetector {
  def detect(rdd: RDD[(Int, String)]): RDD[String] = {
    // Hard code for now, dynamically generate later when I have more sprites
    println("loading sprites")
    println(s"rdd size: ${rdd.count}")
    val normalSprites: List[OpenCVMat] = List(
      OpenCVMat(Imgcodecs.imread(getClass.getResource("/sprites/SuperMarioBros-Nes/0.png").getPath)),
      OpenCVMat(Imgcodecs.imread(getClass.getResource("/sprites/SuperMarioBros-Nes/10.png").getPath)),
      OpenCVMat(Imgcodecs.imread(getClass.getResource("/sprites/SuperMarioBros-Nes/11.png").getPath)),
      OpenCVMat(Imgcodecs.imread(getClass.getResource("/sprites/SuperMarioBros-Nes/12.png").getPath)),
      OpenCVMat(Imgcodecs.imread(getClass.getResource("/sprites/SuperMarioBros-Nes/13.png").getPath)),
      OpenCVMat(Imgcodecs.imread(getClass.getResource("/sprites/SuperMarioBros-Nes/14.png").getPath)),
      OpenCVMat(Imgcodecs.imread(getClass.getResource("/sprites/SuperMarioBros-Nes/15.png").getPath)),
      OpenCVMat(Imgcodecs.imread(getClass.getResource("/sprites/SuperMarioBros-Nes/16.png").getPath)),
      OpenCVMat(Imgcodecs.imread(getClass.getResource("/sprites/SuperMarioBros-Nes/17.png").getPath)),
      OpenCVMat(Imgcodecs.imread(getClass.getResource("/sprites/SuperMarioBros-Nes/18.png").getPath)),
      OpenCVMat(Imgcodecs.imread(getClass.getResource("/sprites/SuperMarioBros-Nes/19.png").getPath)),
      OpenCVMat(Imgcodecs.imread(getClass.getResource("/sprites/SuperMarioBros-Nes/1.png").getPath)),
      OpenCVMat(Imgcodecs.imread(getClass.getResource("/sprites/SuperMarioBros-Nes/20.png").getPath)),
      OpenCVMat(Imgcodecs.imread(getClass.getResource("/sprites/SuperMarioBros-Nes/21.png").getPath)),
      OpenCVMat(Imgcodecs.imread(getClass.getResource("/sprites/SuperMarioBros-Nes/22.png").getPath)),
      OpenCVMat(Imgcodecs.imread(getClass.getResource("/sprites/SuperMarioBros-Nes/23.png").getPath)),
      OpenCVMat(Imgcodecs.imread(getClass.getResource("/sprites/SuperMarioBros-Nes/24.png").getPath)),
      OpenCVMat(Imgcodecs.imread(getClass.getResource("/sprites/SuperMarioBros-Nes/25.png").getPath)),
      OpenCVMat(Imgcodecs.imread(getClass.getResource("/sprites/SuperMarioBros-Nes/26.png").getPath)),
      OpenCVMat(Imgcodecs.imread(getClass.getResource("/sprites/SuperMarioBros-Nes/27.png").getPath)),
      OpenCVMat(Imgcodecs.imread(getClass.getResource("/sprites/SuperMarioBros-Nes/28.png").getPath)),
      OpenCVMat(Imgcodecs.imread(getClass.getResource("/sprites/SuperMarioBros-Nes/29.png").getPath)),
      OpenCVMat(Imgcodecs.imread(getClass.getResource("/sprites/SuperMarioBros-Nes/2.png").getPath)),
      OpenCVMat(Imgcodecs.imread(getClass.getResource("/sprites/SuperMarioBros-Nes/30.png").getPath)),
      OpenCVMat(Imgcodecs.imread(getClass.getResource("/sprites/SuperMarioBros-Nes/31.png").getPath)),
      OpenCVMat(Imgcodecs.imread(getClass.getResource("/sprites/SuperMarioBros-Nes/32.png").getPath)),
      OpenCVMat(Imgcodecs.imread(getClass.getResource("/sprites/SuperMarioBros-Nes/33.png").getPath)),
      OpenCVMat(Imgcodecs.imread(getClass.getResource("/sprites/SuperMarioBros-Nes/34.png").getPath)),
      OpenCVMat(Imgcodecs.imread(getClass.getResource("/sprites/SuperMarioBros-Nes/3.png").getPath)),
      OpenCVMat(Imgcodecs.imread(getClass.getResource("/sprites/SuperMarioBros-Nes/4.png").getPath)),
      OpenCVMat(Imgcodecs.imread(getClass.getResource("/sprites/SuperMarioBros-Nes/5.png").getPath)),
      OpenCVMat(Imgcodecs.imread(getClass.getResource("/sprites/SuperMarioBros-Nes/6.png").getPath)),
      OpenCVMat(Imgcodecs.imread(getClass.getResource("/sprites/SuperMarioBros-Nes/7.png").getPath)),
      OpenCVMat(Imgcodecs.imread(getClass.getResource("/sprites/SuperMarioBros-Nes/8.png").getPath)),
      OpenCVMat(Imgcodecs.imread(getClass.getResource("/sprites/SuperMarioBros-Nes/9.png").getPath))
    )

    println("reflecting sprites")
    val mirrorSprites: List[OpenCVMat] = normalSprites.map(s => {
      val mirrorSprt = OpenCVMat(s.clone())
      Core.flip(mirrorSprt, mirrorSprt, 1)
      mirrorSprt
    })
    val sprites = normalSprites ++ mirrorSprites


    println("generating masks")
    val masks = sprites.zipWithIndex.map(s => {
      val mask: OpenCVMat = OpenCVMat(Mat.zeros(s._1.rows(), s._1.cols(), s._1.`type`()))
      s._1.copyTo(mask)
      val maskColor: Array[Int] = Array(143, 39, 146)

      for (
        i <- 0 until s._1.rows();
        j <- 0 until s._1.cols()
      ) {
        val c = mask.get(i, j)
        if(c != null && c(0) == maskColor(0) && c(1) == maskColor(1) && c(2) == maskColor(2)) {
          mask.put(i, j, 0, 0, 0)
        } else {
          mask.put(i, j, 255, 255, 255)
        }
      }
      mask
    }).zip(sprites)

    rdd.map({
      case (i: Int, src: String) =>
        println(s"detecting sprites: $src")
        val image = OpenCVMat(Imgcodecs.imread(src))
        // Generate a list of heatmap like images denoting where the corresponding template (sprite) is in the source image.
        // This block causes serialization to fail
        val tmplLocMat = masks.map(sprt_mask => {
          val out = new OpenCVMat()
          Imgproc.matchTemplate(image, sprt_mask._2, out, Imgproc.TM_CCORR_NORMED, sprt_mask._1)
          out
        })

        // For each "heatmap" get the location of the maximum value, which corresponds to the sprite's most probable location
        val maxLocs = tmplLocMat map(tmplHeatMap => {
          Core.minMaxLoc(tmplHeatMap)
        })
        // We know mario is only going to be in the image once, so we can pop off the single maximum value, and return
        // generated bounding box
        val spriteMaxLoc: (Core.MinMaxLocResult, OpenCVMat) = maxLocs.zip(sprites).maxBy {
          case (mmlr: Core.MinMaxLocResult, _: OpenCVMat) =>
            mmlr.maxVal
        }


        val filename = s"/home/mcsmash/dev/deep_thought/data_tools/preprocessing/detect-sprites/output/$i.png"
        // Draw a rectangle around the Mario sprite if the location confidence is above a certain threshold
        if (spriteMaxLoc._1.maxVal > 0.95) {
          Imgproc.rectangle(image, spriteMaxLoc._1.maxLoc, new Point(spriteMaxLoc._1.maxLoc.x + spriteMaxLoc._2.cols(),
            spriteMaxLoc._1.maxLoc.y + spriteMaxLoc._2.rows()), new Scalar(255, 255, 255))
        }
        Imgcodecs.imwrite(filename, image)
        filename
    })
  }
}
