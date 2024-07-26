# Importation des bibliothèques
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.stat import Correlation
import pyspark.sql.functions as F
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd


# Présentation des variables de notre dataset
# date (string; datetime of data)
# timestamp (int; number of seconds since beginning of day)
# day_of_week (int; 0 [monday] - 6 [sunday])
# is_weekend (int; 0 or 1) [boolean, if 1, it's either saturday or sunday, otherwise 0]
# is_holiday (int; 0 or 1) [boolean, if 1 it's a federal holiday, 0 otherwise]
# temperature (float; degrees fahrenheit)
# is_start_of_semester (int; 0 or 1) [boolean, if 1 it's the beginning of a school semester, 0 otherwise]
# month (int; 1 [jan] - 12 [dec])
# hour (int; 0 - 23)


# Création d'une session Spark en local
spark = SparkSession.builder \
    .appName("Gym Attendance Analysis") \
    .master("local[*]") \
    .config("spark.driver.host", "localhost") \
    .getOrCreate()


# Lecture du fichier CSV dans un DataFrame
df = spark.read.csv("hdfs:///user/waren/projethadoop/data.csv", header=True, inferSchema=True)


# Affichage des 50 premières lignes du DataFrame pour vérifier nos données
df.show(5)


## Nettoyage des données

# Verifions le type des données
df.printSchema()

# Création de la colonne days
df = df.withColumn(
    "jours",
    when(df.day_of_week == 0, "Lundi")
    .when(df.day_of_week == 1, "Mardi")
    .when(df.day_of_week == 2, "Mercredi")
    .when(df.day_of_week == 3, "Jeudi")
    .when(df.day_of_week == 4, "Vendredi")
    .when(df.day_of_week == 5, "Samedi")
    .when(df.day_of_week == 6, "Dimanche")
)

timestamp2 = "timestamp2"


# Création de la colonne timestamp au format voulue
df = df.withColumn(timestamp2, to_timestamp(col("date"), "yyyy-MM-dd HH:mm:ssXXX"))

# Création de la colonne date modifiée au format jour/mois/année
df = df.withColumn("date_modif", date_format(col(timestamp2), "d/M/yyyy"))

# Création de la colonne heure modifiée au format heure-minute-seconde
df = df.withColumn("heure_modif", date_format(col(timestamp2), "HH:mm:ss"))

# Création de la colonne décalage horaire
df = df.withColumn("decalage_horaire", date_format(col(timestamp2), "XXX"))

# Création de la colonne mois
df = df.withColumn(
    "mois",
    when(df.month == 1, "Janvier")
    .when(df.month == 2, "Février")
    .when(df.month == 3, "Mars")
    .when(df.month == 4, "Avril")
    .when(df.month == 5, "Mai")
    .when(df.month == 6, "Juin")
    .when(df.month == 7, "Juillet")
    .when(df.month == 8, "Août")
    .when(df.month == 9, "Septembre")
    .when(df.month == 10, "Octobre")
    .when(df.month == 11, "Novembre")
    .when(df.month == 12, "Décembre")
)

# df.withColumn(
#     "month",
#     when(col("month") == 1, lit("Janvier"))
#     .otherwise(
#         when(col("month") == 2, lit("Fevrier"))
#         .otherwise(
#             when(col("month") == 3, lit("Mars"))
#         )
#     )
# )

# Création d'une colonne année
df = df.withColumn("annee", year(col(timestamp2)))

# Création d'une colonne 'temperature_celsius'
df = df.withColumn("temperature_celsius", round((col("temperature") - 32) * 5.0 / 9.0, 2))

# Affichons les 10 premières lignes
df.show(10)

# Création d'une vue temporaire
df.createOrReplaceTempView("gym_attendance")


#########################################################################################################################################


## Passons à l'étape de data visualisation

# Nombre de lignes de notre dataframe df
nbre_de_lignes = df.count()
print(f"Le nombre de lignes dans notre DataFrame est de : {nbre_de_lignes}")

# Nombre total de visites par jour
nb_visites_par_jour = spark.sql("""
    SELECT jours, SUM(number_people) AS nb_total_visites
    FROM gym_attendance
    GROUP BY jours
    ORDER BY nb_total_visites DESC
""")

# df = df\
#     .groupby(col("jours"))\
#     .agg(sum(col("number_people")).alias("nb_total_visites"))\
#     .orderby(col("nb_total_visites"), desc = True)

nb_visites_par_jour.show()


# Jour avec la fréquentation maximale
frequentation_max = nb_visites_par_jour.first()
print("Le jour avec la fréquentation maximale:", frequentation_max['jours'], "avec", frequentation_max['nb_total_visites'], "visites au total")


# Jour avec la fréquentation minimale
frequentation_min = nb_visites_par_jour.orderBy("nb_total_visites").first()
print("Le jour avec la fréquentation minimale:", frequentation_min['jours'], "avec", frequentation_min['nb_total_visites'], "visites au total")


################################################################################################################################

annee_recente = spark.sql("""
    SELECT annee, SUM(number_people) AS nb_total_visites
    FROM gym_attendance
    GROUP BY annee
    ORDER BY annee DESC 
""")

annee_recente.show()


# Nombre total de visites
visites_total = spark.sql("""
    SELECT SUM(number_people) AS nb_total_visites 
    FROM gym_attendance
""")
visites_total.show()


# Fréquentation moyenne par jour
visites_moyennes_par_jour = spark.sql("""
    SELECT jours, AVG(number_people) AS moyennes_par_jour 
    FROM gym_attendance
    GROUP BY jours
    ORDER BY moyennes_par_jour DESC
""")
visites_moyennes_par_jour.show()


# Fréquentation médiane par jour
visites_mediane_par_jour = spark.sql("""
    SELECT jours, percentile_approx(number_people, 0.5) AS mediane_par_jour 
    FROM gym_attendance
    GROUP BY jours
    ORDER BY mediane_par_jour DESC               
""")
visites_mediane_par_jour.show()


################################################################################################################################


# Calculer le nombre total de visites par mois avec SQL
visites_par_mois = spark.sql("""
    SELECT mois, SUM(number_people) AS nb_total_visites
    FROM gym_attendance
    GROUP BY mois
""")


# Le mois le plus fréquenté
mois_plus_visités = spark.sql("""
    SELECT mois, SUM(number_people) AS nb_total_visites
    FROM gym_attendance
    GROUP BY mois
    ORDER BY nb_total_visites DESC
    LIMIT 1
""").first()


# Le mois le moins fréquenté
mois_moins_visités = spark.sql("""
    SELECT mois, SUM(number_people) AS nb_total_visites
    FROM gym_attendance
    GROUP BY mois
    ORDER BY nb_total_visites ASC
    LIMIT 1
""").first()


print(f"Le mois le plus fréquenté est: {mois_plus_visités['mois']} avec {mois_plus_visités['nb_total_visites']} visites")
print(f"Le mois le moins fréquenté est: {mois_moins_visités['mois']} avec {mois_moins_visités['nb_total_visites']} visites")


###############################################################################################################################


# Fréquentation totale par mois pour l'année 2015
frequentation_mensuelle_2015 = spark.sql("""
    SELECT mois, SUM(number_people) AS nb_total_visites_2015
    FROM gym_attendance
    WHERE annee = 2015
    GROUP BY mois
    ORDER BY nb_total_visites_2015 DESC
""")

frequentation_mensuelle_2015.show()


###############################################################################################################################


# Comparaison de la fréquentation pendant et en dehors des vacances

# Fréquentation moyenne pendant les vacances
vacances = spark.sql("""
    SELECT 'Holiday' AS Period, SUM(number_people) AS nb_total_visites, AVG(number_people) AS Visites_Moyenne
    FROM gym_attendance
    WHERE is_holiday = 1
""")


# Fréquentation moyenne en dehors des vacances
hors_vacances = spark.sql("""
    SELECT 'Non-Holiday' AS Period, SUM(number_people) AS nb_total_visites, AVG(number_people) AS Visites_Moyenne
    FROM gym_attendance
    WHERE is_holiday = 0
""")


# Je fusionne les deux DataFrames pour comparer
comparison_df = vacances.union(hors_vacances)
comparison_df.show()


##############################################################################################################################


# Fréquentation associée à chaque température et la moyenne de fréquentation pour chaque température
temperature_stats = spark.sql("""
    SELECT temperature_celsius, SUM(number_people) AS Nbre_total_de_visites, AVG(number_people) AS Nbre_moyen_de_visites
    FROM gym_attendance
    GROUP BY temperature_celsius
    ORDER BY Nbre_total_de_visites DESC
""")

temperature_stats.show(15)


##############################################################################################################################


# Fréquentation moyenne par heure
average_hourly_visits = spark.sql("""
    SELECT hour, AVG(number_people) AS Frequentation_moyenne
    FROM gym_attendance
    GROUP BY hour
    ORDER BY Frequentation_moyenne DESC
""")

# Afficher les résultats
average_hourly_visits.show(23)


#########################################################################################################################################


# Graphiques

# Sélection de la colonne hour en integer
df = df.withColumn("hour", df["hour"].cast("integer"))

# Groupons les données par heure et comptons le nombre de visites pour chaque heure
visites_par_heure = df.groupBy("hour").count()

# Trions les données par heure pour une meilleure visualisation
visites_par_heure = visites_par_heure.orderBy("hour")

# Convertissons en Pandas DataFrame pour la visualisation
pandas_df = visites_par_heure.toPandas()

# Génération du graphique de barres
plt.figure(figsize=(14, 7))
bars = plt.bar(pandas_df['hour'], pandas_df['count'], color='green')

# Graphique en barres
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()*0.05, yval, int(yval), va='bottom')
plt.title('Distribution des visites par heure')
plt.xlabel('Heure de la journée')
plt.ylabel('Nombre de visites')
plt.xticks(pandas_df['hour'])
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.show()


###############################################################################################################################

# Nous commençons par exclure les jours sans visites
df_filtered = df.filter(df.number_people > 1)

# Groupons par nombre de personnes et comptons les occurrences
nb_visites = df_filtered.groupBy("number_people").count()

# Convertissons en unn Pandas DataFrame pour la visualisation
pandas_df = nb_visites.toPandas()
pandas_df.sort_values("number_people", inplace=True)

# Générons le graphique de barres
plt.figure(figsize=(12, 6))
plt.bar(pandas_df['number_people'], pandas_df['count'], color='green')
plt.title('Fréquence du nombre de personnes')
plt.xlabel('Nombre de personnes')
plt.ylabel('Frequence')
plt.show()

##############################################################################################################################

# Gardons les colonnes nécessaires pour notre matrice de corrélation
colonnes = ['number_people','temperature', 'is_holiday', 'is_start_of_semester', 'month']
assembler = VectorAssembler(inputCols=colonnes, outputCol="features")
df_vecteurs = assembler.transform(df).select("features")

# Calculons notre matrice de corrélation
# Correlation de spearman au cas où les relations entre les variables ne sont pas linéaires
matrice_correlation = Correlation.corr(df_vecteurs, "features","spearman").head()

# Extraire la matrice de corrélation en tant que tableau Numpy
corr_array = matrice_correlation[0].toArray()

# Affichons la matrice de corrélation
print("Matrice de corrélation:")
print(corr_array)

# Création de notre correlation heatmap
plt.figure(figsize=(12, 6))
sns.heatmap(corr_array, annot=True, fmt=".2f", cmap='coolwarm', xticklabels=colonnes, yticklabels=colonnes)
plt.title("Matrice de corrélation")
plt.show()

###############################################################################################################################

# Gardons les colonnes nécessaires
data = df.select("temperature_celsius", "number_people").collect()

# Préparation des axes pour notre graphique
temperatures = [row['temperature_celsius'] for row in data]
people_counts = [row['number_people'] for row in data]

# Création du graphique
plt.figure(figsize=(10, 6))
plt.scatter(temperatures, people_counts, alpha=0.5)
plt.title("Distribution du nombre de personnes en fonction de la température")
plt.xlabel("Temperature (°C)")
plt.ylabel("Nombre de personnes")
plt.grid(True)
plt.show()

###############################################################################################################################

# Arrêter la session Spark
spark.stop()

