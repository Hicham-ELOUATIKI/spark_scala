package paristech

import org.apache.spark.SparkConf
import org.apache.spark.sql.functions.{sum, lower, split, explode}
import org.apache.spark.sql.{DataFrame, SparkSession}


object WordCount {

  def main(args: Array[String]): Unit = {

    val conf = new SparkConf().setAll(Map(
      "spark.scheduler.mode" -> "FIFO",
      "spark.speculation" -> "false",
      "spark.reducer.maxSizeInFlight" -> "48m",
      "spark.serializer" -> "org.apache.spark.serializer.KryoSerializer",
      "spark.kryoserializer.buffer.max" -> "1g",
      "spark.shuffle.file.buffer" -> "32k",
      "spark.default.parallelism" -> "12",
      "spark.sql.shuffle.partitions" -> "12"
    ))

    val spark = SparkSession
      .builder
      .config(conf)
      .appName("TP spark : Word Count")
      .getOrCreate()

    import spark.implicits._

    val sc = spark.sparkContext

    spark.sparkContext.setLogLevel("WARN")
    /** ******************************************************************************
      *
      * TP 1
      *        - Lecture de données
      *        - Word count , Map Reduce
      * *******************************************************************************/

    // a)
    val dataDir = System.getProperty("user.dir") + "/data"

    val rdd = sc.textFile(dataDir+"/README.md" )
    // b)
    println("5 first rows of the RDD")
    rdd.take(5).foreach(println)
    println("Fin d'execution  b")
    // c)
    val wordCount = rdd
      .flatMap { line: String => line.split(" ") }
      .map { word: String => (word, 1) }
      .reduceByKey { (i: Int, j: Int) => i + j }
      .toDF("word", "count")
    wordCount.show()
    println("Fin d'execution c")
    // d)

    wordCount.orderBy($"count".desc).show()
    println("Fin d'execution d")
    // e)
    val df_lower = wordCount.withColumn("word_lower", lower($"word"))
    df_lower.show()
    println("Fin d'execution e")

    // f)
    val df_grouped = df_lower
      .groupBy("word_lower")
      .agg(sum("count").as("new_count"))

    df_grouped.orderBy($"new_count".desc).show()
    println("Fin d'execution f version 1")

    // Une Autre version de l'exercice, en n'utilisant que les dataFrame et des operations sur les colonnes
    println("With dataFrame only")

    val df2 = spark
      .read
      .text(dataDir+"/README.md") // read.text retourne un dataFrame avec une seule colonne, nommée "value"
      .withColumn("words", split($"value", " "))
      .select("words")
      .withColumn("words", explode($"words"))
      .withColumn("words", lower($"words"))
      .groupBy("words")
      .count
    println("Fin d'execution f version 1")
    df2.orderBy($"count".desc)show()

    // ou encore, de façon plus concise:
   val df3 = spark
      .read
      .text(dataDir+"/README.md") // read.text retourne un dataFrame avec une seule colonne, nommée "value"
      .withColumn("words", explode(split($"value", " ")))
      .withColumn("words", lower($"words"))
      .groupBy("words")
      .count
      .orderBy($"count".desc)
      .show()
    println("Fin d'execution f version 2")

    // ----------------- word count ------------------------

    // Plusieurs exemples de syntaxes, de la plus lourde à la plus légère.
    // Préférez la deuxième syntaxe: les types assurent la consistence des données, et les noms de variables permettent
    // le lire le code plus facilement.

    val df_wordCount = sc.textFile(dataDir+"/README.md" )
      .flatMap { case (line: String) => line.split(" ") }
      .map { case (word: String) => (word, 1) }
      .reduceByKey { case (i: Int, j: Int) => i + j }
      .toDF("word", "count")

    df_wordCount.orderBy($"count".desc).show()


    val df_wordCount_light = sc.textFile(dataDir+"/README.md" )
      .flatMap { line: String => line.split(" ") }
      .map { word: String => (word, 1) }
      .reduceByKey { (i: Int, j: Int) => i + j }
      .toDF("word", "count")

    df_wordCount_light.orderBy($"count".desc).show()


    val df_wordCount_lighter = sc.textFile(dataDir+"/README.md" ) // output RDD of lines : RDD[String]
      .flatMap(line => line.split(" ")) // output RDD of words : RDD[String]
      .map(word => (word, 1)) // output RDD of (Key, Value) pairs : RDD[(String, Int)]
      .reduceByKey((i, j) => i + j) // output RDD of (Key, ValueTot) pairs, where ValueTot is the sum of all value associated with the Key
      .toDF("word", "count") // transform RDD to DataFrame with columns names "word" and "count"

    df_wordCount_lighter.orderBy($"count".desc).show()


    val df_wordCount_lightest = sc.textFile(dataDir+"/README.md" )
      .flatMap(_.split(" "))
      .map((_, 1))
      .reduceByKey(_ + _)
      .toDF("word", "count")
    df_wordCount_lightest.orderBy($"count".desc).show()
  }
}
