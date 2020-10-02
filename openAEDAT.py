import os
import cv2
import sys
import struct
import numpy as np
from scipy import  signal
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.models import model_from_json
from filterUtils import filterUtils

class aedatUtils:


    def loadaerdat(datafile='path.aedat', length=0, version="aedat", debug=1, camera='DVS128'):
        # constants
        aeLen = 8  # 1 AE event takes 8 bytes
        readMode = '>II'  # struct.unpack(), 2x ulong, 4B+4B
        td = 0.000001  # timestep is 1us
        if(camera == 'DVS128'):
            xmask = 0x00fe  # Bin -> 0000 0000 1111 1110 || Dec -> 254
            xshift = 1
            ymask = 0x7f00  # Bin -> 0111 1111 0000 0000 || Dec -> 32512
            yshift = 8
            pmask = 0x1     # Bin -> 0000 0000 0000 0001 || Dec -> 1
            pshift = 0
        else:
            raise ValueError("Unsupported camera: %s" % (camera))

        aerdatafh = open(datafile, 'rb')
        k = 0  # line number
        p = 0  # pointer, position on bytes
        statinfo = os.stat(datafile)
        if length == 0:
            length = statinfo.st_size # Define 'length' = Tamanho do arquivo

        print("file size", length)
        
        # Verifica a versão do Python. 
        if sys.version[0] == '3':
            value = 35 # Se for >= 3 le o cabeçalho em binário.
        else:
            value = '#' # Se for < 3 le o cabeçalho como string.

        # header
        lt = aerdatafh.readline()
        while lt and lt[0] == value:
            p += len(lt)
            k += 1
            lt = aerdatafh.readline() 
            if debug >= 2:
                print(str(lt))
            continue
        
        # variables to parse
        timestamps = []
        xaddr = []
        yaddr = []
        pol = []
        
        # read data-part of file
        aerdatafh.seek(p)
        s = aerdatafh.read(aeLen)
        p += aeLen
        
        print(xmask, xshift, ymask, yshift, pmask, pshift)    
        while p < length:
            addr, ts = struct.unpack(readMode, s)
            # parse event type
            if(camera == 'DVS128'):     
                x_addr = (addr & xmask) >> xshift # Endereço x -> bits de 1-7
                y_addr = (addr & ymask) >> yshift # Endereço y -> bits de 8-14
                a_pol = (addr & pmask) >> pshift  # Endereço polaridade -> bit 0            
                if debug >= 3: 
                    print("ts->", ts) 
                    print("x-> ", x_addr)
                    print("y-> ", y_addr)
                    print("pol->", a_pol)

                timestamps.append(ts)
                xaddr.append(x_addr)
                yaddr.append(y_addr)
                pol.append(a_pol)
                    
            aerdatafh.seek(p)
            s = aerdatafh.read(aeLen)
            p += aeLen        

        if debug > 0:
            try:
                print("read %i (~ %.2fM) AE events, duration= %.2fs" % (len(timestamps), len(timestamps) / float(10 ** 6), (timestamps[-1] - timestamps[0]) * td))
                n = 5
                print("showing first %i:" % (n))
                print("timestamps: %s \nX-addr: %s\nY-addr: %s\npolarity: %s" % (timestamps[0:n], xaddr[0:n], yaddr[0:n], pol[0:n]))
            except:
                print("failed to print statistics")
        t, x, y, p = np.array(timestamps), np.array(xaddr), np.array(yaddr), np.array(pol)
        return t - t[0], x, y, p



    def matrix_active(x, y, pol,filtered=None):
    
        matrix = np.zeros([128, 128]) # Cria uma matriz de zeros 128x128 onde serão inseridos os eventos
        pol = (pol - 0.5) # Os eventos no array de Polaridade passam a ser -0.5 ou 0.5
        
        if(len(x) == len(y)): # Verifica se o tamanho dos arrays são iguais   
            for i in range(len(x)):
                val = 0
                #se a flag do filtro for true. Os eventos serão somados
                #para que eles sejam normalizados pelo maior valor de um acumulo de eventos
                #e depois retirados por um limiar de ~30%
                if filtered == None or filtered == False:
                    val = pol[i]
                elif filtered == True:
                    val = 1
                matrix[x[i], y[i]] += val # insere os eventos dentro da matriz de zeros
        else:
            print("error x,y missmatch")    

        if filtered:
            maxValue = matrix.max()
            matrix = matrix/maxValue
            #matrix[matrix <= 0.5] = 0
            #matrix[np.logical_and(matrix > 0.1, matrix <= 0.3)] = 0.1
            #matrix[matrix >= 0.5] = 1
            matrix = (matrix * 255) # Normaliza a matriz para 8bits -> 0 - 255
        else:
            idx = 0
            limiar = 0.5
            for i in matrix: # Limita os eventos em dentro do limiar
                for j, v in enumerate(i):
                    if v > limiar:
                        matrix[idx][j] = limiar
                    if v < (limiar-1):
                        matrix[idx][j] = (limiar-1)
                idx += 1
            if limiar != 1:
                matrix = (matrix * 255) + 127.5 # Normaliza a matriz para 8bits -> 0 - 255
            
        return matrix

    def getFrameTimeBased(timeArray, polArray, xPosArray, yPosArray,timeStamp, Ti):
        aux = 0
        t2 = timeArray[(timeArray > Ti) & (timeArray <= Ti + timeStamp)]
        x2 = xPosArray[aux : aux + len(t2)]
        y2 = yPosArray[aux : aux + len(t2)]
        p2 = polArray[aux : aux + len(t2)]
        aux += len(t2)
        img = matrix_active(x2, y2, p2)
        img = rotateMatrix(img)
        return img

    def getFramesTimeBased(timeArray, polArray, xPosArray, yPosArray,timeStamp,filtered=None):
        totalImages = []
        i, aux = 0, 0
        images = []
        
        while (i + timeStamp) < abs(timeArray[-1]):
            t2 = timeArray[(timeArray > i) & (timeArray <= i + timeStamp)]
            x2 = xPosArray[aux : aux + len(t2)]
            y2 = yPosArray[aux : aux + len(t2)]
            p2 = polArray[aux : aux + len(t2)]
            aux += len(t2)
            img = aedatUtils.matrix_active(x2, y2, p2,filtered)
            rotacao = aedatUtils.rotateMatrix(img)
            images.append(img)	
            i += timeStamp
        totalImages.extend(images)
        totalImages = np.array(totalImages)
        return totalImages


    def rotateMatrix(mat): 
        N = len(mat)
        # Consider all squares one by one 
        for x in range(0, int(N/2)): 
            
            # Consider elements in group    
            # of 4 in current square 
            for y in range(x, N-x-1): 
                
                # store current cell in temp variable 
                temp = mat[x][y] 
    
                # move values from right to top 
                mat[x][y] = mat[y][N-1-x] 
    
                # move values from bottom to right 
                mat[y][N-1-x] = mat[N-1-x][N-1-y] 
    
                # move values from left to bottom 
                mat[N-1-x][N-1-y] = mat[N-1-y][x] 
    
                # assign temp to left 
                mat[N-1-y][x] = temp 

        return mat
    

    

    def loadDataDemo():
        #read data
        path = '/home/eduardo/Documentos/DVS/Eduardo work/Mestrado/Datasource/AEDAT_files/random data/shorter records/multi_objects_2.aedat'
        t, x, y, p = aedatUtils.loadaerdat(path)
        promediar = False
        tI=100000 #10 ms
        # tI=10000 #10 ms

        totalImages = []
        totalImages = aedatUtils.getFramesTimeBased(t,p,x,y,tI,False)

        #plot
        fig,axarr = plt.subplots(1)
        handle = None
        if promediar == True:
            maxProMediacao = 10
            indexProMediacao = 0
            imagemPromediada = np.zeros([128,128])
            for f in totalImages:
                f = f.astype(np.uint8)
                f[f==0] = 1
                f[f==255] = 1
                f[f==127] = 0
                imagemPromediada += f
                indexProMediacao += 1
                if indexProMediacao == 10:
                    indexProMediacao = 0
                    imagemPromediada = imagemPromediada/10
                    imagemPromediada = imagemPromediada*255
                    imagemPromediada = imagemPromediada.astype(np.uint8)
                    if handle is None:
                        handle = plt.imshow(np.dstack([imagemPromediada,imagemPromediada,imagemPromediada]))
                        #handle = plt.imshow(imagemPromediada)
                    else:
                        handle = plt.imshow(np.dstack([imagemPromediada,imagemPromediada,imagemPromediada]))
                        #handle.set_data(imagemPromediada)
                    imagemPromediada = np.zeros([128,128])
                    plt.pause(0.01)
                    plt.draw()
        else:
            for f in totalImages:
                f = f.astype(np.uint8)
                #f = filterUtils.sobelFilter(f)
                if handle is None:
                    handle = plt.imshow(np.dstack([f,f,f]))
                else:
                    handle = plt.imshow(np.dstack([f,f,f]))

                plt.pause(0.01)
                plt.draw()
       


    def main(objClass=None, tI=50000, split=False, size=0.20):
        if objClass == None:
            t, x, y, p = loadaerdat("/home/user/GitHub/Classification_DVS128/aedatFiles/" + input("Nome do arquivo:") + ".aedat")
        else:
            objClass = objClass.split(", ")
            totalImages = []
            labels = []
            for j, v in enumerate(objClass):
                t, x, y, p = loadaerdat("/home/user/GitHub/Classification_DVS128/aedatFiles/" + str(v) + ".aedat")
                i, aux = 0, 0
                images = []
                while (i + tI) < t[-1]:
                    t2 = t[(i < t) & (t <= i + tI)]
                    x2 = x[aux : aux + len(t2)]
                    y2 = y[aux : aux + len(t2)]
                    p2 = p[aux : aux + len(t2)]
                    aux += len(t2)
                    images.append(matrix_active(x2, y2, p2))	
                    labels.append([j])
                    i += tI
                totalImages.extend(images)
                
            totalImages, labels = np.array(totalImages), np.array(labels)
            
            randomize = np.arange(len(labels))
            np.random.shuffle(randomize)
            totalImages = totalImages[randomize]
            labels = labels[randomize]
            
            if split:
                totalImages_train, totalImages_test, labels_train, labels_test = train_test_split(totalImages, labels, test_size=size, random_state=42)
                return totalImages_train, totalImages_test, labels_train, labels_test
            else:
                return totalImages, labels
        
        
if __name__ == "__main__":
	aedatUtils.loadDataDemo()