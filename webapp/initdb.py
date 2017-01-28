import sqlite3, os
conn = sqlite3.connect('reviews.sqlite')
c=conn.cursor()
c.execute('CREATE TABLE review_db (review TEXT, sentiment INTEGER, date TEXT)')
conn.commit()
conn.close()    