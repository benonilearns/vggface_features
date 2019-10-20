import cv2
import imutils
import numpy as np
import os
import json
from PIL import Image
import psycopg2
from psycopg2.extensions import AsIs
from psycopg2.extensions import register_adapter
from psycopg2.extensions import adapt
from psycopg2.extras import Json
from psycopg2.extras import DictCursor
import sqlalchemy
import requests

temp_dir = "/media/benoni/800x600/"
face_dir = "/media/benoni/face_cropped/"

def get_records():
        try:
                connection = psycopg2.connect(user="benoni", password="foobar", host="127.0.0.1", port="1111", database="foobar")
                cursor = connection.cursor()
                query = 'SELECT face_rectangle,photo_id,"800x640" FROM "ORDER_144k"'
                cursor.execute(query)
                records = [r for r in cursor.fetchall()]
                return records
        except (Exception, psycopg2.Error) as error :
                print ("Error while fetching data from PostgreSQL", error)

def dlAndCrop(record):
    face_rectangle = record[0]
    photo_id = record[1]
    url = record[2]
    try:
        r = requests.get(url, stream=True)
        if r.status_code == 200:
            with open (temp_dir + photo_id + ".jpg", 'wb') as f:
                f.write(r.content)
                cropFace((temp_dir + photo_id + ".jpg"), face_rectangle, photo_id)
    except Exception as e:
        print(e)

def cropFace(image, face_rectangle, photo_id):
    photo = cv2.imread(image, 3)
    rectangle = json.loads(face_rectangle)
    height = rectangle['height']
    width = rectangle['width']
    top = rectangle['top']
    left = rectangle['left']
    crop_img = photo[top:top+height, left:left+width]
    resized = cv2.resize(crop_img,(224,224))
    cv2.imwrite(face_dir + photo_id + "_224x224.jpg", resized)

if __name__ == '__main__':
    records = get_records()
    goal = len(records)
    count = 0
    for record in records:
        dlAndCrop(record)
        count += 1
        if count % 1000 == 0:
            print("Done: " + str(count))
            print("To go: " + str(goal - count))
