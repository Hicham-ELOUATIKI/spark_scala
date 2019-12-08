# Spark project MS Big Data Télécom : Kickstarter campaigns

Spark project for MS Big Data Telecom based on Kickstarter campaigns 2019-2020


## TP3

**Versioning**
* Scala : 2.11.11
* Spark : 2.4.4 (changed in build.sbt)
###Modèle entraîné 
###Avec la grille

F1-score obtained on test data : 0.6045  

    +--------------+--------------+------+  
    |final_status  |predictions   |count |  
    +--------------+--------------+------+  
    |      1       |     0.0      |  988 |  
    |      0       |     1.0      | 3405 |  
    |      1       |     1.0      | 2406 |  
    |      0       |     0.0      | 3944 |  
    +--------------+--------------+------+  

###Sans la grille
F1-score obtained on test data: 0.627

    +--------------+--------------+------+  
    |final_status  |predictions   |count |  
    +--------------+--------------+------+  
    |      1       |     0.0      | 1717 |  
    |      0       |     1.0      | 2375 |  
    |      1       |     1.0      | 1677 |  
    |      0       |     0.0      | 4974 |  
    +--------------+--------------+------+  



