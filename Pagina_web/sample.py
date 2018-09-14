
import sqlite3
import random

con = sqlite3.connect('DataBase.db')
c = con.cursor()


def crear_tabla():
    c.execute("CREATE TABLE IF NOT EXISTS Base_de_datos(Accuracy REAL, Prediction TEXT, Reality TEXT)")


def entrada_dinamica_datos():
    emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    acu = random.randrange(50, 100)
    pred = random.choice(emotions)
    real = random.choice(emotions)


    c.execute("INSERT INTO Base_de_datos (Accuracy, Prediction, Reality)"
              " VALUES (?, ?, ?)", (acu, pred, real))

    con.commit()

crear_tabla()

for i in range(30):
    entrada_dinamica_datos()

c.close()
con.close()
