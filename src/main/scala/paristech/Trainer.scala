package paristech

import org.apache.spark.SparkConf
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.feature.{RegexTokenizer, CountVectorizer, IDF, StopWordsRemover, StringIndexer, OneHotEncoder, VectorAssembler}
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.tuning.{ParamGridBuilder, TrainValidationSplit}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator


object Trainer {

  def main(args: Array[String]): Unit = {

    val conf = new SparkConf().setAll(Map(
      "spark.scheduler.mode" -> "FIFO",
      "spark.speculation" -> "false",
      "spark.reducer.maxSizeInFlight" -> "48m",
      "spark.serializer" -> "org.apache.spark.serializer.KryoSerializer",
      "spark.kryoserializer.buffer.max" -> "1g",
      "spark.shuffle.file.buffer" -> "32k",
      "spark.default.parallelism" -> "12",
      "spark.sql.shuffle.partitions" -> "12",
      "spark.driver.maxResultSize" -> "2g"
    ))

    val spark = SparkSession
      .builder
      .config(conf)
      .appName("TP Spark : Trainer")
      .getOrCreate()
    spark.sparkContext.setLogLevel("error")

    /*******************************************************************************
      *
      *       TP 3
      *
      *       - lire le fichier sauvegarder précédemment
      *       - construire les Stages du pipeline, puis les assembler
      *       - trouver les meilleurs hyperparamètres pour l'entraînement du pipeline avec une grid-search
      *       - Sauvegarder le pipeline entraîné
      *
      *       if problems with unimported modules => sbt plugins update
      *
      ********************************************************************************/

    val dataDir = System.getProperty("user.dir") + "/data"

    // 1 - Charger le dataframe

    val df = spark.read.parquet(dataDir + "/prepared_trainingset/")

    // 2 - Utiliser les données textuelles

    // a - 1er stage: La première étape est de séparer les textes en mots (ou tokens) avec un tokenizer.

    val tokenizer = new RegexTokenizer()
      .setPattern("\\W+")
      .setGaps(true)
      .setInputCol("keywords")
      .setOutputCol("tokens")

    // b - 2e stage: On veut retirer les stop words pour ne pas encombrer le modèle avec des mots qui ne véhiculent pas de sens.
    val remover = new StopWordsRemover()
      //.setStopWords(stopwords)
      .setInputCol(tokenizer.getOutputCol)
      .setOutputCol("removed")

    // c - 3e stage: La partie TF de TF-IDF est faite avec la classe CountVectorizer.
    val vectorizer = new CountVectorizer()
      .setInputCol(remover.getOutputCol)
      .setOutputCol("vectorized")

    // d - 4e stage: Trouvez la partie IDF. On veut écrire l’output de cette étape dans une colonne “tfidf”.
    val idf = new IDF()
      .setInputCol(vectorizer.getOutputCol)
      .setOutputCol("tfidf")

    // 3 - Convertir les catégories en données numériques

    // e - 5e stage: Convertir la variable catégorielle “country2” en quantités numériques.
    val indexer_country = new StringIndexer()
      .setInputCol("country2")
      .setOutputCol("country_indexed")

    // f - 6e stage: Convertir la variable catégorielle “currency2” en quantités numériques.
    val indexer_currency = new StringIndexer()
      .setInputCol("currency2")
      .setOutputCol("currency_indexed")

    // g - 7e stage & 8e stage: transformer ces deux catégories avec un “one-hot encoder” en créant  les colonnes “currency_onehot” et “country_onehot”
    val encoder_country = new OneHotEncoder()
      .setInputCol("country_indexed")
      .setOutputCol("country_onehot")

    val encoder_currency = new OneHotEncoder()
      .setInputCol("currency_indexed")
      .setOutputCol("currency_onehot")

    // 4 - Mettre les données sous une forme utilisable par Spark.ML

    // h - 9e stage: Assembler les features "tfidf", "days_campaign", "hours_prepa", "goal", "country_onehot", "currency_onehot"  dans une seule colonne “features”.
    val vecAssembler = new VectorAssembler()
      .setInputCols(Array("tfidf", "days_campaign", "hours_prepa", "goal", "country_onehot", "currency_onehot"))
      .setOutputCol("features")

    // i - 10e stage: Le modèle de classification
    val lr = new LogisticRegression()
      .setElasticNetParam(0.0)
      .setFitIntercept(true)
      .setFeaturesCol("features")
      .setLabelCol("final_status")
      .setStandardization(true)
      .setPredictionCol("predictions")
      .setRawPredictionCol("raw_predictions")
      .setThresholds(Array(0.7, 0.3))
      .setTol(1.0e-6)
      .setMaxIter(300)
   //   .setMaxIter(20)

    // j - Enfin, créer le pipeline en assemblant les 10 stages définis précédemment, dans le bon ordre.
    val stages = Array(tokenizer, remover, vectorizer, idf, indexer_country, indexer_currency, encoder_country, encoder_currency, vecAssembler, lr)

    val pipeline = new Pipeline().setStages(stages)


    // 5 - Entraînement et tuning du modèle

    // k - Créer un dataFrame nommé “training” et un autre nommé “test”  à partir du dataFrame
    // chargé initialement de façon à le séparer en training et test sets dans les proportions 90%, 10% respectivement.
    val Array(training, test) = df.randomSplit(Array[Double](0.9, 0.1), 1)

    // l - Préparer la grid-search pour satisfaire les conditions explicitées ci-dessus
    //puis lancer la grid-search sur le dataset “training” préparé précédemment.

    val paramGridBuilder: ParamGridBuilder = new ParamGridBuilder()
    // Pour activer la grille changer false par true
    if(true) {
      paramGridBuilder.addGrid(lr.regParam, Array(10e-2, 10e-4, 10e-6, 10e-8))
        .addGrid(vectorizer.minDF, 55.0 to 95.0 by 20.0)
    }
    val paramGrid = paramGridBuilder.build()
    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("final_status")
      .setPredictionCol("predictions")
      .setMetricName("f1")

    val trainValidationSplit = new TrainValidationSplit()
      .setEstimator(pipeline)
      .setEvaluator(evaluator)
      .setEstimatorParamMaps(paramGrid)
      .setTrainRatio(0.7)

    val model_train = trainValidationSplit.fit(training)

    // m - Appliquer le meilleur modèle trouvé avec la grid-search aux données test.
    // Mettre les résultats dans le dataFrame df_WithPredictions. Afficher le f1-score du modèle sur les données de test.

    val df_WithPredictions = model_train.transform(test)
      .select("features", "final_status", "predictions")

    val F1Score = evaluator.evaluate(df_WithPredictions)

    println("F1-score obtained on test data: " + BigDecimal(F1Score).setScale(4, BigDecimal.RoundingMode.HALF_UP).toDouble)

    // n - Afficher df_WithPredictions.groupBy("final_status", "predictions").count.show()

    df_WithPredictions.groupBy("final_status", "predictions").count().show()

    // m - sauvegarde le modéle entraîné entraîné
    model_train.write.overwrite().save(dataDir + "/model_train")
  }
}
