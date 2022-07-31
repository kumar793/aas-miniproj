def my_first_api():
    import PySimpleGUI as sg
    from collections import Iterable
    import numpy as np
    import imutils
    import pickle
    import time
    import cv2
    import csv
    import os
    import os.path
    from sklearn.preprocessing import LabelEncoder
    from sklearn.svm import SVC
    import pickle
    from imutils import paths
    from pandas.core.frame import DataFrame
    import datetime,gspread,random
    import mysql.connector
    import oauth2client
    from oauth2client.service_account import ServiceAccountCredentials
    import pandas as pd 

    conn = mysql.connector.connect(user='root', password='kumar', host='127.0.0.1', database='scsvmv',auth_plugin='mysql_native_password')
    scope = ["https://spreadsheets.google.com/feeds",'https://www.googleapis.com/auth/spreadsheets',"https://www.googleapis.com/auth/drive.file","https://www.googleapis.com/auth/drive"]

    creds = ServiceAccountCredentials.from_json_keyfile_name("credentials.json", scope)
    client = gspread.authorize(creds)
    sheet = client.open("face recognization1").sheet1
    def flatten(lis):
                for item in lis:
                    if isinstance(item, Iterable) and not isinstance(item, str):
                        for x in flatten(item):
                            yield x
                    else:
                        yield item

    class attendance():
        def __init__(self):
            self.embeddingFile = "output/embeddings.pickle" #initial name for embedding file
            self.embeddingModel = "openface_nn4.small2.v1.t7" #initializing model for embedding Pytorch
            self.prototxt = "model/deploy.prototxt"
            self.model =  "model/res10_300x300_ssd_iter_140000.caffemodel"
            self.cascade = 'haarcascade_frontalface_default.xml'
            self.detector = cv2.dnn.readNetFromCaffe(self.prototxt, self.model)
            self.embedder = cv2.dnn.readNetFromTorch(self.embeddingModel)
            self.recognizerFile = "output/recognizer.pickle"
            self.labelEncFile = "output/le.pickle"
            

        def register(self):
            self.Name = ' '
            self.Roll_Number = ''
            layout = [  [sg.Text('Enter the name and rollnumber')],
                                [sg.Text("NAME", size =(15, 1), font=16),sg.InputText(key='-name-', font=16)],
                                [sg.Text("Roll_no", size =(15, 1), font=16),sg.InputText(key='-roll-no-', font=16)],
                                [sg.Button("Run"), sg.Button('Exit')] ]
            window = sg.Window('register here', layout)

            while True:        
                    event, values = window.Read()
                    if event == "Exit" or event == sg.WIN_CLOSED:
                        break
                    if event == "Run":
                            
                            Name = values['-name-']
                            Roll_Number = values['-roll-no-']
                            window.close()
            detector = cv2.CascadeClassifier(self.cascade)
            self.dataset = 'dataset'
            sub_data = self.Name
            path = os.path.join(self.dataset, sub_data)

            if not os.path.isdir(path):
                os.mkdir(path)
                print(sub_data)

            info = [str(self.Name), str(self.Roll_Number)]
            with open('student.csv', 'a') as csvFile:
                write = csv.writer(csvFile)
                write.writerow(info)
                csvFile.close()

            sg.theme("LightGreen")

            # Define the window layout
            layout = [
                    [sg.Text("dataset creation", size=(60, 1), justification="center")],
                    [sg.Image(filename="", key="-IMAGE-")],
                    [sg.Button("Exit", size=(10, 1))]
                
            ]

            # Create the window and show it without the plot
            window = sg.Window("dataset creation", layout, location=(800, 400))

            print("Starting video stream...")
            cam = cv2.VideoCapture(0)
            time.sleep(2.0)
            total = 0

            while total < 30:
                event, values = window.read(timeout=20)
                if event == "Exit" or event == sg.WIN_CLOSED:
                    break
                    
                print(total)
                _, frame = cam.read()
                img = imutils.resize(frame, width=600)
                rects = detector.detectMultiScale(
                cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), scaleFactor=1.1,
                                    minNeighbors=5, minSize=(30, 30))

                for (x, y, w, h) in rects:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    p = os.path.sep.join([path, "{}.png".format(
                                            str(total).zfill(5))])
                    cv2.imwrite(p, img)
                    total += 1

                    imgbytes = cv2.imencode(".png", frame)[1].tobytes()
                    window["-IMAGE-"].update(data=imgbytes)
            window.close()

        def preprocess(self):
            imagePaths = list(paths.list_images(self.dataset))
            knownEmbeddings = []
            knownNames = []
            total = 0
            conf = 0.5

                    #we start to read images one by one to apply face detection and embedding
            for (i, imagePath) in enumerate(imagePaths):
                print("Processing image {}/{}".format(i + 1,len(imagePaths)))
                name = imagePath.split(os.path.sep)[-2]
                image = cv2.imread(imagePath)
                image = imutils.resize(image, width=600)
                (h, w) = image.shape[:2]
                #converting image to blob for dnn face detection
                imageBlob = cv2.dnn.blobFromImage(
                            cv2.resize(image, (300, 300)), 1.0, (300, 300),(104.0, 177.0, 123.0), swapRB=False, crop=False)

                            #setting input blob image
                self.detector.setInput(imageBlob)
                            #prediction the face
                detections = self.detector.forward()

                if len(detections) > 0:
                    i = np.argmax(detections[0, 0, :, 2])
                    confidence = detections[0, 0, i, 2]

                    if confidence > conf:
                        #ROI range of interest
                        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                        (startX, startY, endX, endY) = box.astype("int")
                        face = image[startY:endY, startX:endX]
                        (fH, fW) = face.shape[:2]

                        if fW < 20 or fH < 20:
                            continue
                        #image to blob for face
                        faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255, (96, 96), (0, 0, 0), swapRB=True, crop=False)
                        #facial features embedder input image face blob
                        self.embedder.setInput(faceBlob)
                        vec = self.embedder.forward()
                        knownNames.append(name)
                        knownEmbeddings.append(vec.flatten())
                        total += 1

            print("Embedding:{0} ".format(total))
            data = {"embeddings": knownEmbeddings, "names": knownNames}
            f = open(self.embeddingFile, "wb")
            f.write(pickle.dumps(data))
            f.close()
            print("Process Completed")
            print("click back for live attendance")
        def train(self):
            

            print("Loading face embeddings...")
            data = pickle.loads(open(self.embeddingFile, "rb").read())

            print("Encoding labels...")
            labelEnc = LabelEncoder()
            labels = labelEnc.fit_transform(data["names"])


            print("Training model...")
            self.recognizer = SVC(C=1.0, kernel="linear", probability=True)
            self.recognizer.fit(data["embeddings"], labels)

            f = open(self.recognizerFile, "wb")
            f.write(pickle.dumps(self.recognizer))
            f.close()

            f = open(self.labelEncFile, "wb")
            f.write(pickle.dumps(labelEnc))
            f.close()
            print("model trained")

    class test(attendance):
        def __init__(self):
            super().__init__()
        def test(self):
            self.recognizer = pickle.loads(open(self.recognizerFile, "rb").read())
            self.le = pickle.loads(open(self.labelEncFile, "rb").read())
            
            print("[INFO] starting video stream...")
            cam = cv2.VideoCapture(0)
            time.sleep(2.0)

            while True:
            
                _, frame = cam.read()
                frame = imutils.resize(frame, width=600)
                (h, w) = frame.shape[:2]
                imageBlob = cv2.dnn.blobFromImage(
                    cv2.resize(frame, (300, 300)), 1.0, (300, 300),
                    (104.0, 177.0, 123.0), swapRB=False, crop=False)

                self.detector.setInput(imageBlob)
                detections = self.detector.forward()
                conf = 0.9995
                self.le = pickle.loads(open(self.labelEncFile, "rb").read())

                for i in range(0, detections.shape[2]):

                    confidence = detections[0, 0, i, 2]

                    if confidence > conf:

                        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                        (startX, startY, endX, endY) = box.astype("int")

                        face = frame[startY:endY, startX:endX]
                        (fH, fW) = face.shape[:2]

                        if fW < 20 or fH < 20:
                            continue

                        faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255, (96, 96), (0, 0, 0), swapRB=True, crop=False)
                        self.embedder.setInput(faceBlob)
                        vec = self.embedder.forward()

                        preds = self.recognizer.predict_proba(vec)[0]
                        j = np.argmax(preds)
                        proba = preds[j]
                        name = self.le.classes_[j]
                        with open('student.csv', 'r') as csvFile:
                            reader = csv.reader(csvFile)
                            for row in reader:
                                box = np.append(box, row)
                                name = str(name)
                                if name in row:
                                    person = str(row)
                                    print(name)
                            listString = str(box)
                            print(box)
                            if name in listString:
                                singleList = list(flatten(box))
                                listlen = len(singleList)
                                Index = singleList.index(name)
                                name = singleList[Index]
                                Roll_Number = singleList[Index + 1]
                                


                        text = "{} : {} : {:.2f}%".format(name, Roll_Number, proba * 100)
                        y = startY - 10 if startY - 10 > 10 else startY + 10
                        cv2.rectangle(frame, (startX, startY), (endX, endY),
                                    (0, 0, 255), 2)
                        cv2.putText(frame, text, (startX, y),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
                cv2.imshow("Frame", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    
                    break

            cam.release()
            cv2.destroyAllWindows()

    def enroll(name,Roll_Number,d,t):
        nrows = len(sheet.col_values(1))
        pin=random.randint(999,9999)
        sheet.update_cell(nrows+1,1,name)
        sheet.update_cell(nrows+1,2,Roll_Number)
        sheet.update_cell(nrows+1,3,d)
        sheet.update_cell(nrows+1,4,t)



    class liverecognization(test,):


        def __init__(self):
            super().__init__()
            
            self.recognizer = pickle.loads(open(self.recognizerFile, "rb").read())
            self.le = pickle.loads(open(self.labelEncFile, "rb").read())
        
        def liverecognization(self):
            conf = 0.8
            cam = cv2.VideoCapture(0)
            sg.theme("LightGreen")

            # Define the window layout
            layout = [
                [sg.Text("Attendnace live", size=(60, 1), justification="center")],
                [sg.Image(filename="", key="-IMAGE-")],
                [sg.Button("Take attendance",size = (15,1))],
                [sg.Button("Exit", size=(10, ))],
                
            ]

            # Create the window and show it without the plot
            window = sg.Window("Automatic attendnace", layout, location=(0,0), size=(900,600), keep_on_top=True )

            cap = cv2.VideoCapture(0)
            
            time.sleep(2.0)
        

            while True:
                event, values = window.read(timeout=20)
                if event == "Exit" or event == sg.WIN_CLOSED:
                    break
            
                        
                _, frame = cap.read()
            
                frame = imutils.resize(frame, width=600)
                (h, w) = frame.shape[:2]
                imageBlob = cv2.dnn.blobFromImage(
                        cv2.resize(frame, (300, 300)), 1.0, (300, 300),
                        (104.0, 177.0, 123.0), swapRB=False, crop=False)

                self.detector.setInput(imageBlob)
                detections = self.detector.forward()

                for i in range(0, detections.shape[2]):

                        confidence = detections[0, 0, i, 2]

                        if confidence > conf:

                                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                                (startX, startY, endX, endY) = box.astype("int")

                                face = frame[startY:endY, startX:endX]
                                (fH, fW) = face.shape[:2]

                                if fW < 20 or fH < 20:
                                        continue

                                faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255, (96, 96), (0, 0, 0), swapRB=True, crop=False)
                                self.embedder.setInput(faceBlob)
                                vec = self.embedder.forward()

                                preds = self.recognizer.predict_proba(vec)[0]
                                j = np.argmax(preds)
                                proba = preds[j]
                                name = self.le.classes_[j]
                                with open('student.csv', 'r') as csvFile:
                                        reader = csv.reader(csvFile)
                                        for row in reader:
                                                box = np.append(box, row)
                                                name = str(name)
                                                if name in row:
                                                        person = str(row)
                                                        print(name)
                                                listString = str(box)
                                                print(box)
                                                names =[]
                                                roll_no =[]
                                            
                                                
                                                if name in listString:
                                                        singleList = list(flatten(box))
                                                        listlen = len(singleList)
                                                        Index = singleList.index(name)
                                                        name = singleList[Index]
                                                        Roll_Number = singleList[Index + 1]
                                                    
                                                                        
                                        text = "{} : {} : {:.2f}%".format(name, Roll_Number, proba * 100)
                                        y = startY - 10 if startY - 10 > 10 else startY + 10
                                        cv2.rectangle(frame, (startX, startY), (endX, endY),
                                                        (12, 140, 255), 2)
                                        cv2.putText(frame, text, (startX, y),
                                                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

                                        if event =="Take attendance":
                                            now=datetime.datetime.now()                                      
                                            d=now.strftime('%y/%m/%d').replace('/0','/')
                                            t=now.strftime('%H:%M:%S')
                                            enroll(name,Roll_Number,d,t)                                                 
                                            sg.popup_auto_close('Mr/ms ',name,'dont worry ','your attendance updated sucessfully')
                                            
                                            try:
                                                mySql_insert_query = """INSERT INTO student (sname,srollnumber,pdate,ptimr) 
                                                VALUES (%s,%s,%s,%s) """
                                                name = str(name)
                                                Roll_Number = str(Roll_Number)
                                                cursor = conn.cursor()
                                                val = (name,Roll_Number,d,t)
                                                cursor.execute(mySql_insert_query,val)
                                                conn.commit()
                                                cursor.close()
                                                sg.popup_auto_close(cursor.rowcount, "Record inserted successfully into student table")
                                                
                                            except:
                                                sg.popup_auto_close("already attendance given")
                imgbytes = cv2.imencode(".png", frame)[1].tobytes()
                window["-IMAGE-"].update(data=imgbytes)
            window.close()
    a = liverecognization()
    sg.theme("lightgreen")
    layout = [
            [sg.Text("registration form", size=(100, 1), justification="center")],
                [sg.Button("register", size=(100, 1))],
                    [sg.Button("test", size=(100, 1))],
                    [sg.Button("attendance", size=(100, 1))],
                    [sg.Button("current entries", size=(100, 1))],
                    [sg.Button("registered students", size=(100, 1))],
                    [sg.Button("track student", size=(100, 1))],
            [sg.Button("Exit", size=(100, 1))],
                    
            ]
    window = sg.Window("MENU", layout)

    # Create an event loop
    while True:
        event, values = window.read()
        # End program if user closes window or
        # presses the OK button
        if event == "Exit" or event == sg.WIN_CLOSED:
            break
        elif event == "register":
            
            a.register()
            a.preprocess()
            a.train()
            
        elif event == "test":
            return{a.test()} 
        elif event == "attendance":
            return{a.liverecognization()}
        elif event == "current entries":
            q = "select * from student"
            df = pd.read_sql_query(q,conn)
            sg.Print(df)
            
        elif event == "registered students":
            df = pd.read_csv("student.csv")
            sg.Print(df)


        '''elif event == "track student":
            rno = ''
            layout1 = [  [sg.Text('Enter the  rollnumber')],
                                [sg.Text("Rollno", size =(15, 1), font=16),sg.InputText(key='-roll1-no-', font=16)],
                                [sg.Button("Run"), sg.Button('Exit')] ]
            window = sg.Window('track student', layout1)

            while True:        
                    event, values = window.Read()
                    if event == "Exit" or event == sg.WIN_CLOSED:
                        break
                    if event == "Run":
                            rno = values['-roll1-no-']
                            window.close()
            q ="select * from student where srollnumber = %s "
            
            val = (rno)
        
            df = pd.read_sql_query(q,val,conn)
            sg.Print(df)'''

            

    window.close()

my_first_api()