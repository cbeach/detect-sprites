package org.apind.detectsprites

import org.apache.spark.{SparkConf, SparkContext}
import org.apind.util.OpenCVMat
import org.opencv.core._
import org.opencv.core.Scalar
import org.opencv.imgcodecs.Imgcodecs
import org.opencv.imgproc.Imgproc


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
  //Runner.runWithoutSpark()
}

/**
  * Use this when submitting the app to a cluster with spark-submit
  * */

object DetectSpritesApp {
  def main(args: Array[String]) {
    // spark-submit command should supply all necessary config elements
    Runner.run(new SparkConf())
  }
}

object Runner {
  def run(conf: SparkConf): Unit = {
    System.loadLibrary(Core.NATIVE_LIBRARY_NAME)
    val sc = new SparkContext(conf)
    val frames: List[(Int, String)] = (1 until 99).map(frame_number => {
      (frame_number, getClass.getResource(s"/test-examples/SuperMarioBros-Nes/$frame_number.png").getPath)
    }).toList

    val rdd = sc.makeRDD(frames)
    TemplateBasedSpriteDetector.detect(rdd)
    sc.stop()
  }

  def runWithoutSpark(): Unit = {
    System.loadLibrary(Core.NATIVE_LIBRARY_NAME)
    val frames: List[(Int, String)] = (1 until 99).map(frame_number => {
      (frame_number, getClass.getResource(s"/test-examples/SuperMarioBros-Nes/$frame_number.png").getPath)
    }).toList

    println("loading sprites")
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
    val mirrorSprites: List[OpenCVMat] = normalSprites.map(s => {
      val mirrorSprt = OpenCVMat(s.clone())
      Core.flip(mirrorSprt, mirrorSprt, 1)
      mirrorSprt
    })

    val sprites = normalSprites ++ mirrorSprites
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
      //val filename = s"/home/mcsmash/dev/deep_thought/data_tools/preprocessing/detect-sprites/output/mask/${s._2}.png"
      //Imgcodecs.imwrite(filename, mask)
      mask
    }).zip(sprites)

    frames.map({
      case (i: Int, src: String) =>
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
            //println(s"src: $src")
            //println(s"maxLoc: ${mmlr.maxLoc}, maxVal: ${mmlr.maxVal}")
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
