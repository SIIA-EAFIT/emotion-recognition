from flask import Flask, render_template, request
import sqlite3 as sql

app = Flask(__name__)


@app.route("/", methods=['GET', 'POST'])
def inicio():
    return render_template('Main.html')


@app.route("/Descripcion")
def montaje():
    return render_template('Descripcion.html')


@app.route("/Enlaces")
def enlaces():
    return render_template('Enlaces.html')


@app.route("/resultados", methods=['GET'])
def resultados():

    con = sql.connect("DataBase.db")
    con.row_factory = sql.Row

    cur = con.cursor()

    cur.execute("SELECT Accuracy, Prediction, Reality FROM Base_de_datos"
                " ORDER BY Accuracy DESC LIMIT 25")

    rows = cur.fetchall()

    return render_template('Resultados.html', rows=rows)

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=8000)
