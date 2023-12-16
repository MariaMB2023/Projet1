# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
#import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
#from datetime import datetime as dt
#import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
#from imblearn.over_sampling import RandomOverSampler
#from sklearn.compose import ColumnTransformer
#from sklearn.impute import SimpleImputer
#from plotly.subplots import make_subplots
import plotly.express as px
import joblib
#from sklearn.metrics import f1_score

df=pd.read_csv('C:/Users/famil/ProjetKickStarter/Kickstarter Campaigns DataSet.csv')



st.sidebar.title('Sommaire')

pages = ['Présentation','Exploration des Données','Analyse des données','Modélisation','Application','Conclusion']
page = st.sidebar.radio('allez vers la page', pages)


#***************************************************************************************************
#***************************************************************************************************
# PAGE 0 Présentation
#***************************************************************************************************
#***************************************************************************************************

if page == pages[0]:
    st.title("Prediction du succès d'une campagne de financement participatif")
    st.write(" Contexte du Projet")
    st.write("Une campagne participative ou crowdfunding c’est un moyen de collecte de fonds en ligne pour soutenir le développement d’un projet, qu’il s’agisse de la création d’une entreprise, d’un documentaire, d’un film, etc.")
    st.write("Arrêtons-nous au plus grand succès du crowfunding « star citizen » un jeu vidéo avec 300 millions de dollar récolté ou la marque « Humble » avec une campagne réussi sur un smoothie protéiné. On peut se demander ce qui a fait le succès de leur campagne.")
    st.write("Le financement alternatif, tel que le crowfunding en ligne, est souvent plus attrayant qu’un prêt bancaire pour concrétiser une idée. Toutefois, la réalisation d’une campagne de crowdfunding est cruciale pour déterminer si un projet va décoller ou non.")
    st.write("L'objectif de ce projet est de comprendre les clés de la réussite d’une campagne de financement participatif, en classant les projets en réussite ou échec, puis en analysant les raisons potentielles de ces résultats.")
    st.write("Cela permettra de guider les créateurs dans la mise en place de leur campagne et de prendre des décisions éclairées concernant son lancement.")
    st.image("Crowfounding2.jpg")

#***************************************************************************************************
#***************************************************************************************************
# PAGE 1 Exploration des données

#***************************************************************************************************
#***************************************************************************************************

elif page ==pages[1]:
    st.title("  Exploration des données")
    st.write("Le jeu de données utilisé a été obtenu grâce au webscraping de Kickstarter provenant de webro-bots.io ( https://webrobots.io/kickstarter-datasets/). Les données ont été combinées, prétraitées et mise à disposition dans un jeu de données contenant des projets(campagnes) de 2009 à 2020. Ce jeu de données est disponible sur le site https://www.kaggle.com/datasets/yashkantharia/kickstarter-campaigns-dataset-20")
    
    st.write("Aperçu des données")
    st.dataframe(df.head())
    
    st.write("Dimensions initial du Dataframe:")
    st.write(df.shape)
    st.write("Type des données:")
    st.write(df.dtypes)
    
    # nombre des données par status
    
    functions_to_apply = {
        'id' : lambda x: len(x)
    }
    testStatus=df.groupby('status').agg(functions_to_apply)
    st.write("nombre des données par status:")
    st.write(testStatus)
    
    df = df[(df['status']=='successful') | (df['status']=='failed')]
    st.write("Total de campagnes successful et failed:")
    st.write(len(df))
    
    
    # valeurs manquantes

    if st.checkbox("Aficher les valeurs manquantes"):
       st.dataframe(df.isna().sum())

    # examiner les doublons

    #print('nombre de doublons: ',df.duplicated().sum(),'\n')

    # éliminer la colonne Unnamed: 0

    df.drop(columns=["Unnamed: 0"],inplace=True)
    if st.checkbox("Aficher les doublons"):
       st.write(df.duplicated().sum())

    #print('nombre de doublons sans colonne Unnamed: 0 : ',df.duplicated().sum(),'\n')

    # éliminer les doublons

    df = df.drop_duplicates(keep='first') 
    #print('Total de campagnes successful et failed sans doublons: ', len(df),'\n')

    # Changement nom des colonnes

    df = df.rename({'sub_category':'category', }, axis = 1)
    #print(df['category'].unique(),'\n')

    df = df.rename({'main_category':'sub_category'}, axis = 1)
   # print(df['sub_category'].unique(),'\n')

    # changement du type de colonne

    #print(type(df['launched_at'][0]),'\n')

    df['launched_at'] = pd.to_datetime(df['launched_at'])

    #print(type(df['deadline'][0]),'\n')

    df['deadline'] = pd.to_datetime(df['deadline'])

    df['duration'] = df['duration'].astype('int') 

    # création des colonnes

    df['year_launched'] = df['launched_at'].dt.year
    df['month_launched'] = df['launched_at'].dt.month

    df['year_deadline'] = df['deadline'].dt.year
    df['month_deadline'] = df['deadline'].dt.month

    st.write("Traitement fait au niveau des données:")
    st.write("• Élimination des doublons ")
    st.write("•	Changement du nom des colonnes : la colonne « sub_category » par « category » et « main_category » par « sub_category »")
    st.write("•	Changement de type de colonnes : Le type des variables dates(str) ont été modifiés au type date, le type de la variable duration(float) a été modifié au type int")
    st.write("•	Création des colonnes : À partir des colonnes de type dates quatre nouvelles colonnes année et mois ont été créées")
    if st.checkbox("Nombre total des données après Nettoyage:"):
       st.write(len(df))
    
    st.dataframe(df.dtypes)
    st.write("##### Aperçu statistique")
    st.dataframe(df.describe())

#***************************************************************************************************
#***************************************************************************************************
# PAGE 2 ANALYSE DES DONNEES

#***************************************************************************************************
#***************************************************************************************************
       
elif page == pages[2]:
    st.title("  Analyse des données")
    st.write("L’analyse servira à décrire et résumer des données de manière significative afin de tirer des informations.")
    st.write("Il s’agit tout d'abord de savoir quel est le nombre des campagnes réussies et le nombre des campagnes échouées à l’aide de la variable aidant à trouver la réponse « status ». Sur un total de 184899 données il y a 109205 réussies et 75694 échouées.")
    df_clean = pd.read_csv("df_Cleaning.csv")
     
    st.write("Pourcentage des campagnes par status")
    st.image("CampagneStatus.jpg")
 
    
    st.write("L’analyse portera sur 4 axes : •	Catégorie •	Finance •Géographique •Temps ")
    
# Radio button pour les axes
#----------------------------

    page_names = ['Catégorie', 'Finance', 'Géographique', 'Temps']
    page2 = st.radio('Navigation', page_names)

#        Axe Catégorie
#****************************
    
    if page2 == page_names[0]:
        st.subheader("Axe Catégorie")
     # categories par status
     #***********************
     
# DF pour stocker les data de groupby
        df_aggregated = df_clean.groupby(['category', 'status']).size().unstack(fill_value=0).reset_index()

# Calculer les pourcentages
        df_aggregated['total'] = df_aggregated['successful'] + df_aggregated['failed']
        df_aggregated['success_percentage'] = round((df_aggregated['successful'] / df_aggregated['total']) * 100,2)
        df_aggregated['failed_percentage'] = round((df_aggregated['failed'] / df_aggregated['total']) * 100,2)

# charger palette seaborn
        sns.set()

# Récupérer couleur par defaut de seaborn
        seaborn_palette = sns.color_palette()

# Convertir les couleurs en format hexadécimal
        color_discrete_map = {'successful': seaborn_palette[1], 'failed': seaborn_palette[0]}
        color_discrete_map_hex = {k: f'#{int(c[0]*255):02x}{int(c[1]*255):02x}{int(c[2]*255):02x}' for k, c in color_discrete_map.items()}


# Créer graphique interactif avec Plotly Express
        fig2 = px.bar(df_aggregated, x='category', y=['successful', 'failed'],
              labels={'successful': 'Quantité de succès', 'failed': 'Quantité d\'échecs'},
              title='Quantité de succès et d\'échecs par catégorie',
              barmode='group',
              hover_data={'success_percentage': ':.2f%', 'failed_percentage': ':.2f%'},
              color_discrete_map=color_discrete_map_hex)

# Afficher le graphique avec Streamlit
        st.plotly_chart(fig2)
        st.write("                                ")
        st.write("En conclusion, il est incorrect de généraliser en affirmant qu'un plus grand nombre de campagnes entraîne automatiquement un plus grand nombre de réussites, ou qu'un nombre moins élevé de campagnes conduit à un nombre de réussite inférieur. Chaque catégorie de campagne présente ses propres caractéristiques et il est essentiel d'analyser les facteurs spécifiques qui contribuent à la réussite d'une campagne dans chaque contexte")
    
    
    
    # test Anova
    #***************
    
        st.write("Test ANOVA")
        st.write("Ce test nous permettra d'analyser si la catégorie de la campagne a une influence significative sur le nombre de backers qui soutiennent la campagne, ce qui pourrait contribuer à expliquer les résultats observés")
        import statsmodels.api
# est-ce que la catégorie a un effet significatif sur le nombre des backers de la campagne
        st.write("𝐻0: Pas d'effet significatif de la catégorie sur le nombre de backers de la campagne")
        st.write("𝐻1: Il y a un effet significatif de la catégorie de la campagne sur nombre de backers de la campagne")

        result = statsmodels.formula.api.ols('backers_count  ~ category', data=df_clean).fit()
        AnovaTable = statsmodels.api.stats.anova_lm(result)
        st.write(AnovaTable)
        st.write("Conclusion: p-valeur < 𝛼 (0.05), on conclut donc  une influence significative de catégorie sur le nombre de backer")


    #Catégories qui attirent plus de backers
    #*****************************************
        st.image("BplotByCat.jpg")
        st.write("Les catégories qui enregistrent en moyenne ou en médiane le plus grand nombre de 'backers' sont: les jeux (games), les bandes dessinées (comics).")
        st.write("Les catégories qui ont en moyenne ou en médiane le moins de 'backers' sont : les arts manuels (crafts), la nourriture (food), le journalisme (journalism) et la photographie (photography).")
        st.write("Les autres catégories présentent des moyennes ou des médianes de 'backers' qui ne montrent pas de grands écarts entre eux.")
        st.write("La catégorie technologie (technology) attire davantage de 'backers' , mais elle se distingue par un nombre de réussite de campagnes relativement faible. En d'autres termes, bien qu'elle rassemble un grand nombre de 'backers', un nombre significatif de ses campagnes se soldent par un échec.")
 
#         Axe Finance
#***********************************
    elif page2 == page_names[1]:
        st.subheader("Axe Finance")

       
        sns.set_theme()

        
    # Votre fonction de comptage
        def comptage(test):
            suc = len(test[test['status'] == 'successful'])
            fail = len(test[test['status'] == 'failed'])
            return suc, fail

# Calculer les données de comptage
        comp1 = comptage(df_clean[df_clean['goal_usd'] <= 10])
        comp2 = comptage(df_clean[(10 < df_clean['goal_usd']) & (df_clean['goal_usd'] <= 100)])
        comp3 = comptage(df_clean[(100 < df_clean['goal_usd']) & (df_clean['goal_usd'] <= 1000)])
        comp4 = comptage(df_clean[(1000 < df_clean['goal_usd']) & (df_clean['goal_usd'] <= 10000)])
        comp5 = comptage(df_clean[(10000 < df_clean['goal_usd']) & (df_clean['goal_usd'] <= 100000)])
        comp6 = comptage(df_clean[(100000 < df_clean['goal_usd']) & (df_clean['goal_usd'] <= 1000000)])
        comp7 = comptage(df_clean[(1000000 < df_clean['goal_usd']) & (df_clean['goal_usd'] <= 10000000)])
        comp8 = comptage(df_clean[(10000000 < df_clean['goal_usd']) & (df_clean['goal_usd'] <= 100000000)])
        comp9 = comptage(df_clean[(100000000 < df_clean['goal_usd']) & (df_clean['goal_usd'] <= 1000000000)])
        comp = comptage(df_clean[df_clean['goal_usd'] > 1000000000])

# Convertir la palette de couleurs Seaborn en liste hexadécimale
        seaborn_colors = sns.color_palette("tab10", as_cmap=True).colors

# Créer le graphique Plotly
        fig3 = px.bar(x=['<=10', '10-100', '100-1K', '1k-10K', '10K-100K', '100K-1M', '1M-10M', '10M-100M', '100M-1B'],
             y=[comp1[0], comp2[0], comp3[0], comp4[0], comp5[0], comp6[0], comp7[0], comp8[0], comp9[0]],
             labels={'y':'Count', 'x':'goal_usd'},
             title="Campagnes par tranche d'objectif en fonction du statut",
             barmode='group',
             color_discrete_sequence=['orange', 'blue'])
        
        fig3.add_bar(x=['<=10', '10-100', '100-1K', '1k-10K', '10K-100K', '100K-1M', '1M-10M', '10M-100M', '100M-1B'],
            y=[comp1[1], comp2[1], comp3[1], comp4[1], comp5[1], comp6[1], comp7[1], comp8[1], comp9[1]],
            name='failed',
            marker_color='blue')  # Utiliser la deuxième couleur de la palette pour 'failed'

# Afficher le graphique à l'aide de Streamlit
        st.plotly_chart(fig3)

        st.write("En conclusion les campagnes avec un financement entre 1K et 10K dollars affichent des perspectives plus élevées de réussite. De plus, les campagnes sollicitant jusqu’à 1M de dollars ont également la possibilité de rencontrer du succès. En revanche, les campagnes requérant plus d’1M de dollars connaissent une diminution significative de leurs probabilités de réussite.")
        
#             Axe Géographique
#**************************************

    elif page2 == page_names[2]:    
        st.subheader("Axe Géographique")
        st.image("CountryQte.jpg")
        
        
# Charger la palette de couleur de Seaborn
        seaborn_palette = sns.color_palette()

# Convertir les couleurs de Seaborn en format hexadécimal
        def rgb_to_hex(rgb):
            return "#{:02x}{:02x}{:02x}".format(int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255))

        color_sequence = [rgb_to_hex(color) for color in seaborn_palette]

# Créer le graphique avec Plotly Express
        fig4 = px.histogram(df_clean, x='country', color='status', title='Nombre des Campagnes par Pays selon le statut',
                   color_discrete_sequence=color_sequence)
        fig4.update_layout(barmode='group', xaxis_title='Pays', yaxis_title='Nombre de Campagnes')

# Afficher le graphique dans l'application Streamlit
        st.plotly_chart(fig4)
        st.write("Il est indéniable que les États-Unis, le Royaume-Uni, le Canada et l'Australie représentent collecti-vement environ 80% du total de campagnes")
        st.write("possible raisons: Première plateforme de financement participatif, Taille du marché, Langue commune, Écosystème entrepreneurial")

#             Axe Temporelle
#**************************************        
    elif page2 == page_names[3]:
        st.subheader("Axe Temporelle")
        # Charger la palette de couleur de Seaborn
        seaborn_palette = sns.color_palette()

# Convertir les couleurs de Seaborn en format hexadécimal
        def rgb_to_hex(rgb):
            return "#{:02x}{:02x}{:02x}".format(int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255))

        color_sequence = [rgb_to_hex(color) for color in seaborn_palette]

# Créer le graphique avec Plotly Express
        fig5 = px.histogram(df_clean, x='duration', color='status', title='Nombre des Campagnes par  selon le statut',
                   color_discrete_sequence=color_sequence)
        fig5.update_layout(barmode='group', xaxis_title='Duration', yaxis_title='Nombre de Campagnes')
        st.plotly_chart(fig5)
        st.write("La grande majorité des campagnes ont une durée de 30 jours. Cependant, lorsque la durée est étendue à 60 jours, le nombre de campagnes échouées dépassent nettement celui des campagnes réussies, au-delà de cette durée de 60 jours, la quantité des campagnes mises en œuvre diminue notablement.")

#***************************************************************************************************
#***************************************************************************************************
# PAGE 3 MODELISATION

#***************************************************************************************************
#***************************************************************************************************

elif  page == pages[3]:
    from sklearn.metrics import classification_report
#    from sklearn.metrics import confusion_matrix 
    st.title(" Modélisation")
    df_clean = pd.read_csv("df_Cleaning.csv")
    # changement du type de colonne
    df_clean['launched_at'] = pd.to_datetime(df_clean['launched_at'])
     
    # 1.- Feature selection
    df_clean['day_launched'] = df_clean['launched_at'].dt.day
    colonnes =['id', 'currency','blurb','deadline','usd_pledged','creator_id','year_deadline','month_deadline','name', 'slug','launched_at','sub_category','city']

    df_clean = df_clean.drop(colonnes,axis=1)
    
    # 2.- Séparer variable cible et variable explicatives
    #****************************************************

    # target variable cible( status: succesful, failed), feats  variables explicatives
    target = df_clean['status']# y

    feats = df_clean.drop('status', axis=1)# X

    # 3.- séparer les données en jeu de test et jeu d'entrainemment
    #**************************************************************

    X_train, X_test, y_train, y_test = train_test_split(feats, target, test_size=0.25, random_state = 42)


    num_train = X_train[['backers_count', 'blurb_length', 'goal_usd','duration','year_launched','month_launched','day_launched']]
    cat_train = X_train[['country','category'] ]

    num_test = X_test[['backers_count', 'blurb_length', 'goal_usd','duration','year_launched','month_launched','day_launched']]
    cat_test = X_test[['country','category']]   
    
    # 4.- Encodage 
    #*****************************************
    # 4.1 .- encodage des variables catégorielles
    # *****************************************
    # (a) à l'aide de OHE, encoder les variables catégorielles
    # Le paramètre drop permet d'éviter le problème de multicolinéarité

    num_train = num_train.reset_index(drop=True)
    num_test = num_test.reset_index(drop=True)

    #from sklearn.preprocessing import OneHotEncoder

    ohe = OneHotEncoder( drop="first", sparse=False)

    colcat=['country','category']
    cat_train = pd.DataFrame(ohe.fit_transform(cat_train[colcat])) 
    cat_train.columns = ohe.get_feature_names_out(colcat)

    cat_test = pd.DataFrame(ohe.transform(cat_test[colcat]))
    cat_test.columns = ohe.get_feature_names_out(colcat)

    # (b)  reconstituer les jeu train et test en concatenant num_train avec cat_train et num_test avec cat_test
    #dfcat_train = pd.DataFrame(cat_train)
    X_train = pd.concat([num_train,cat_train],axis=1)

    X_test = pd.concat([num_test,cat_test],axis=1)

    # (c) afficher les dimensions des jeux reconstitués
    # 4.2 .- encodage de variables numériques
    # *****************************************
    colonnes= [ 'blurb_length', 'duration','year_launched','month_launched','day_launched']

    scaler = StandardScaler()

    X_train[colonnes] = scaler.fit_transform(X_train[colonnes])

    X_test[colonnes] = scaler.transform(X_test[colonnes])
       
    # 4.3 .- encodage variable cible
    #*********************************
    #  Encoder les modalités de la variable cible status à l'aide d'un LABELENCODER
    # en estimant l'encodage sur le jeu d'entraînement et en l'appliquant sur le jeu d'entraînement et de test.

    # from sklearn.preprocessing import LabelEncoder

    le = LabelEncoder()

    y_train = le.fit_transform(y_train)

    y_test = le.transform(y_test)
    # 4.4.- encodage variables avec outliers
    #************************************

    from sklearn.preprocessing import RobustScaler

    # Soient X_train et X_test, les jeux, respectivement d'entraînement et de test, des variables explicatives

    colonnes=['backers_count','goal_usd']

    scaler = RobustScaler()

    X_train[colonnes] = scaler.fit_transform(X_train[colonnes])

    X_test[colonnes]  = scaler.transform(X_test[colonnes] )
    
 #      importer les modèles entrainés et enregistrés en local
#*************************************************************  
   
    dtc1 = joblib.load("model_dtc")
    rl1 = joblib.load("model_rl")
    rf1 = joblib.load("model_rf")
   # knn1 = joblib.load("model_knn")
    gbc1 = joblib.load("model_gbc")
    
    y_pred_dtc = dtc1.predict(X_test)
    y_pred_rl1 = rl1.predict(X_test)
    y_pred_rf1 = rf1.predict(X_test)
    #y_pred_knn1 = knn1.predict(X_test)
    y_pred_gbc = gbc1.predict(X_test)
    
    mp='#### Meilleurs paramètres '
    vs='#### Score'
    train= 'jeu entrainement: '
    test = 'jeu de test: '
    cr = '#### Classification Report'
    mc = "#### Matrice de Confusion"
    st.image("PerforSansHyper.jpg") 
    st.image("PerforAvecHyper.jpg")
    model_choisi = st.selectbox(label= "Modèle", options = ['DecisionTreeClassifier','Logistic Regression','Random Forest','Gradient Boosting Classifier'])
    if model_choisi == 'DecisionTreeClassifier':
        st.image("arbreDecision.jpg")
        st.write('#### Meilleurs paramètres ') 
        st.write('{max_depth=8, min_samples_leaf=5}')
        # score train et test
        st.write('#### Score')
        st.write('jeu entrainement: ', dtc1.score(X_train,y_train))
        st.write('jeu de test: ', dtc1.score(X_test,y_test))
        y_pred = y_pred_dtc
        #st.write(classification_report (y_test, y_pred))
        st.write('#### Classification Report')
        st.text("Result:\n" + classification_report(y_test, y_pred))
        st.write("#### Matrice de Confusion")
        cmT = pd.crosstab(y_test, y_pred, rownames=['Classe réelle'], colnames=['Classe prédite'])
        st.write(cmT)            
        # Graphique
        fig6, ax = plt.subplots()
        pd.Series(dtc1.feature_importances_, X_train.columns).sort_values(ascending=True).plot(kind='barh', ax=ax, figsize=(4, 8))
        ax.set_title('Decision Tree Feature Importances')
        ax.set_xlabel('Importance')
        ax.set_ylabel('Feature')
        st.pyplot(fig6)
        st.image("TreeFIPourcentage.png")
        st.write("constat:")
        st.write("Les variables 'backers_count' et 'goal_usd' émergent comme des éléments cruciaux dans la classification de notre arbre de décision. Le nombre de contributeurs (backers) s'avère particulièrement déterminant pour le succès d'une campagne. En outre, la variable 'goal_usd' se positionne en deuxième place en termes d'importance. Son influence est clairement perceptible dans la structure de l'arbre, contribuant à la formation de nœuds et de feuilles caractérisés par une classe majoritaire de 'Sucessful'.")
        st.write("backers_count et goal_usd ont determiné à 99%  le status d'une campagne, les autres variables influence seulement à 0.1%")
        st.write("Conclusion: le modèle de Decision Tree Classifier, nous donne des bonnes prédictions")
        
    elif model_choisi == 'Logistic Regression':
        st.write(mp)  
        st.write("{C= 1, penalty='l2', solver= 'liblinear'}")
        st.write(vs)
        st.write(train, rl1.score(X_train,y_train))
        st.write(test, rl1.score(X_test,y_test))
        y_pred =y_pred_rl1
        st.write(cr)
        st.text("Result:\n" + classification_report(y_test, y_pred))
        st.write(mc)
        cmT = pd.crosstab(y_test, y_pred, rownames=['Classe réelle'], colnames=['Classe prédite'])
        st.write(cmT)  
        # Graphique
        fig7, ax = plt.subplots()
        pd.Series(rl1.coef_[0], X_train.columns).sort_values(ascending=True).plot(kind='barh', figsize=(4,8));
        ax.set_title('Logistic Regression Feature Importances')
        ax.set_xlabel('Importance')
        ax.set_ylabel('Feature')
        st.pyplot(fig7)
        
        st.write("constat:")
        st.write("La variable 'goal_usd' ne semble pas jouer un rôle significatif dans la sortie du modèle, ce qui pourrait expliquer, par exemple, pourquoi certaines campagnes avec un objectif financier élevé ne réussissent pas nécessairement, ou pourquoi seulement un nombre limité de campagnes aboutissent à un état de 'Successful'. En revanche, il est évident que le nombre de contributeurs ('backer_count') joue un rôle très important dans la sortie du modèle")
        st.image("RocRL.jpg")
        st.write("Constat:")
        st.write("Si le taux de vrais positifs est égal à 1 et le taux de faux positifs est égal à 1, cela signifie que toutes les campagnes 'Failed' ont été mal prédites. Si les taux de vrais positifs et de faux positifs sont tous deux égaux à 0, cela indique que toutes les campagnes sont déclarées comme 'Failed'. Lorsque la courbe ROC se rapproche du point (0,1), cela suggère que les campagnes 'Successful' et 'Failed' sont bien prédites. Dans notre cas, un AUC de 0.97 correspond à la proportion d'échantillons correctement classés. En d'autres termes, 97% des campagnes ont été correctement classées")
        st.write("Conclusion:")
        st.write(" Notre modèle de Régression logistique effectue déjà des très bonnes prédictions des campagnes 'Sucessful' et 'Failed'")      
        
        
    elif model_choisi == 'Random Forest':
        st.write(mp)
        st.write("{max_depth =14, min_samples_leaf= 5}")
        st.write(vs)
        st.write(train, rf1.score(X_train,y_train))
        st.write(test, rf1.score(X_test,y_test))
        y_pred = y_pred_rf1
        st.write(cr)
        st.text("Result:\n" + classification_report(y_test, y_pred))
        st.write(mc)
        cmT = pd.crosstab(y_test, y_pred, rownames=['Classe réelle'], colnames=['Classe prédite'])
        st.write(cmT) 
        # graphique
        fig8, ax = plt.subplots()
        pd.Series(rf1.feature_importances_, X_train.columns).sort_values(ascending=True).plot(kind='barh', figsize=(4,8));
        ax.set_title('Random Forest Feature Importances')
        ax.set_xlabel('Importance')
        ax.set_ylabel('Feature')
        st.pyplot(fig8)
        st.write("consta: les variables 'backer_count' et 'goal_usd' semblent jouer un rôle significatif dans la sortie du modèle ")
        st.write("conclusion: Le modèle de Random Forest effectue de très bonnes prédictions mais le nombre de Faux Positif est encore élevé")
        
    elif model_choisi == 'Gradient Boosting Classifier':
        st.write("Gradient Boosting: C'est un ensemble de «weak learners» (algorithmes de faible performance) créés les uns après les autres formant un « strong learner » (algorithme beaucoup plus efficace). Chaque « weak learner» est entraîné pour corriger les erreurs des «weak learners» précédents. La Particularité de Gradient Boosting réside dans sa tentative de prédire, à chaque étape, les résidus (écart entre la moyenne et la réalité). Les « weak learners » ont tous autant de poids dans le système de votation peu importe leur performance.")
        st.write(mp)
        st.write("{'learning_rate': 0.1, 'max_depth': 4, 'n_estimators': 250}")
        st.write(vs)
        st.write(train, gbc1.score(X_train,y_train))
        st.write(test, gbc1.score(X_test,y_test))
        y_pred = y_pred_gbc
        st.write(cr)
        st.text("Result:\n" + classification_report(y_test, y_pred))
        st.write(mc)
        cmT = pd.crosstab(y_test, y_pred, rownames=['Classe réelle'], colnames=['Classe prédite'])
        st.write(cmT) 
        import shap
        explainer = shap.TreeExplainer(gbc1)
        shap_values = explainer.shap_values(X_test)
        #shap.summary_plot(shap_values, X_test, plot_type="bar")    
        # Create a SHAP summary plot using Streamlit
        st.set_option('deprecation.showPyplotGlobalUse', False)  # To avoid warning about pyplot use

# Plot the SHAP summary plot
        st.write("#### Importance moyenne des variables sur l'output du Gradient Boosting Classifier")
        shap.summary_plot(shap_values, X_test, plot_type="bar")

# Display the plot using Streamlit
        st.pyplot()
        st.write("### Participation de chaque variable dans la prediction final")
        shap.summary_plot(shap_values, X_test)
        st.pyplot()
        st.write("Les variables 'backer_count' et 'goal_usd' exercent une influence sur l'ensemble des prédictions du modèle.")
        st.write("backer_count:")
        st.write("• Une valeur plus faible ou basse de backer_count est associée à une valeur SHAP négative, ce qui oriente la prédiction vers 0, indiquant ainsi un échec ou un statut 'failed'. • Certaines occurrences de valeurs basses de backer_count génèrent des valeurs SHAP positives, favorisant ainsi une prédiction de 1 (projet réussi ou 'successful').• À mesure que la valeur de backer_count augmente, elle est associée à une valeur SHAP positive, renforçant la prédiction de 1 (projet réussi ou 'successful').")
        st.write("goal_usd: ")
        st.write("•	Une valeur plus élevée de goal_usd est associée à une valeur SHAP négative, incitant à prédire 0, ce qui indique un échec ou un statut 'failed'. • À l'inverse, une valeur plus faible ou basse de goal_usd est liée à une valeur SHAP positive, encourageant ainsi la prédiction de 1, signifiant un projet réussi ou 'successful'.")
   
    
    
#***************************************************************************************************
#***************************************************************************************************

# PAGE 4 APPLICATION

#***************************************************************************************************
#***************************************************************************************************
    
elif  page == pages[4]:
    
    
    st.write("### Testez la Réussite de votre campagne")

    df_clean = pd.read_csv("df_Cleaning.csv")
# changement du type de colonne
    df_clean['launched_at'] = pd.to_datetime(df_clean['launched_at'])
    #st.write(df_clean.dtypes)

# 1.- Feature selection
    df_clean['day_launched'] = df_clean['launched_at'].dt.day
    colonnes =['id', 'currency','blurb','deadline','usd_pledged','creator_id','year_deadline','month_deadline','name', 'slug','launched_at','sub_category','city']

    df_clean = df_clean.drop(colonnes,axis=1)

#***************************************************************************3

  # data interface ui
    goal = st.number_input('Saisissez un montant pour votre campagne',min_value = df_clean['goal_usd'].min()) #, max_value = df_clean['goal_usd'].max()
    st.write('Le montant saisi est: ', goal)
    
    dur = st.number_input('Saisissez le nombre de jours qui durera votre campagne', min_value = df_clean['duration'].min()) # , max_value = df_clean['duration'].max()
    st.write('Le montant saisi est:', dur)
   
    categ = st.selectbox(label= "Choisissez une Catégorie", options =['food', 'publishing' ,'technology' ,'film & video' ,'games' ,'theater',

                                                                 'journalism' ,'music' ,'design', 'art' ,'crafts' ,'photography', 'fashion', 'comics' ,'dance'])
    st.write('La catégorie saisi est: ', categ)
    
    bkT = int(df_clean['backers_count'].median())
    countryT = df_clean['country'].mode()[0]
    bLenT = int(df_clean['blurb_length'].median())
    yearT = int(df_clean['year_launched'].mean())
    monthT = int(df_clean['month_launched'].mean())
    dayT = int(df_clean['day_launched'].mean())
    
    dataSaisi = {'backers_count': [bkT] , 
                'country': [countryT] ,
                'status': [0],
                'category': [categ],
                'blurb_length': [bLenT],
                'goal_usd': [goal],
                'duration': [dur],
                'year_launched': [yearT],
                'month_launched': [monthT],
                'day_launched': [dayT]
                }

    dfSaisi = pd.DataFrame(data=dataSaisi)
    

# 2.- Séparer variable cible et variable explicatives
#****************************************************

# target variable cible( status: succesful, failed), feats  variables explicatives
 #   target = dfSaisi['status']# y

    X_saisi = dfSaisi.drop('status', axis=1)# X

    
    # 3.- séparer les données en jeu de test et jeu d'entrainemment
#**************************************************************
    num_test = X_saisi[['backers_count', 'blurb_length', 'goal_usd','duration','year_launched','month_launched','day_launched']]
    cat_test = X_saisi[['country','category']]
    

# 4.- Encodage 
#*****************************************
# 4.1 .- encodage des variables catégorielles
# *****************************************
# (a) à l'aide de OHE, encoder les variables catégorielles
# Le paramètre drop permet d'éviter le problème de multicolinéarité
# import ohe
    ohe = joblib.load('encoder_joblib')
    

    num_test = num_test.reset_index(drop=True)

    colcat=['country','category']
    cat_test = pd.DataFrame(ohe.transform(cat_test[colcat])) 
    cat_test.columns = ohe.get_feature_names_out(colcat)

    X_saisi = pd.concat([num_test,cat_test],axis=1)

    # 4.2 .- encodage de variables numériques
# *****************************************

    colonnes= [ 'blurb_length', 'duration','year_launched','month_launched','day_launched']

    scaler =joblib.load('scaler_joblib')
    X_saisi[colonnes] = scaler.transform(X_saisi[colonnes])
 

# 4.4.- encodage variables avec outliers
#************************************

    colonnes=['backers_count','goal_usd']

    rscaler = joblib.load('rscaler_joblib')

    X_saisi[colonnes] = rscaler.transform(X_saisi[colonnes])

        
    gbc1H = joblib.load("model_dtc")
    y_pred_gbc=gbc1H.predict(X_saisi)
    #st.write(y_pred_gbc)
    st.write("les probabilité en pourcentage de réussite(1) ou d'echecs(0):")
    resultproba =(gbc1H.predict_proba(X_saisi))*100
    st.write(resultproba) 
    
elif  page == pages[5]:
    st.write("##### Challenges")
    st.write("•	Gestion des formats JSON au début du projet")
    st.write("•	Modification du mentorat en cours d'avancement")
    st.write("•	Charge de travail conséquente pour une seule personne")
    st.write("•	Utilisation de bibliothèques qui ne font pas partie du programme du Data Analyst")
    st.write("##### Bilan")
    st.write("•	Très bonnes performances des modèles")
    st.write("•	Application approfondie des concepts de l'analyse de données et du machine learning, incluant les métriques et l'ajustement des hyperparamètres, poussée jusqu'à la recherche de l'interprétabilité")
    st.write("##### Améliorations")
    st.write("•	Intégration du nom du projet dans les prédictions en utilisant des techniques de traitement du langage naturel (NLP) ou des caractéristiques telles que la longueur du nom du projet")
    st.write("•	Introduction de tests statistiques supplémentaires, par exemple, la vérification de la normalité des variables à l’aide du test de Shapiro-Wilk")

# Ajout du texte de l'auteur
st.sidebar.title('Auteur')

# Ajout du lien LinkedIn
st.sidebar.markdown('[Maria Brenzikofer(Sehgelmeble)](https://www.linkedin.com/in/mariabrenzikofer/)')



