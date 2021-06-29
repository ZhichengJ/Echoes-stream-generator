from datetime import datetime
from fnmatch import fnmatch
import os
# # pip install wavio --user
# pip install PySoundFile --user
# import soundfile as sf
import pandas as pd
import numpy as np

from pyo import *
import time
import matplotlib.pyplot as plt
import librosa
import librosa.display

import subprocess
import os
import urllib.request
import sys

import configparser
import json
import requests

import warnings
warnings.filterwarnings("ignore")


class SoundGenerator():
    def __init__(self, rate=44100, fre=440, seconds_split=1, sps=10, noise_dbfs=12):
        self.rate = int(rate)
        self.fre = int(fre)

        self.noise_dbfs = int(noise_dbfs)

        # self.noise_mu = int(0)
        self.noise_sigma = int(1)

        self.seconds_split = int(seconds_split)
        self.samples_per_second = int(sps)

        self.min_scale = 1
        self.max_scale = self.noise_dbfs * 5

        self.file_ext = '.wav'

    def __generateNoise(self, time, amp):
        noise_sig = self.noise_sigma

        if amp < self.noise_dbfs:
            noise_sig = self.noise_dbfs * 0.2

        return np.random.uniform(-1 * noise_sig, 1 * noise_sig, int(self.rate * time))


    def generate(self, t, s_n, destination_path):
        if len(t) < 2:
            return 'assets/default-noise.wav'
        
        t = [float(i) for i in t]
        s_n = [float(i) for i in s_n]
        # Archivos de config
        config = configparser.ConfigParser()
        config.read('configuracion.properties')

        # Número de muestras de cada uno de los tipos de ecos
        s = int(config.get('Seleccion', 's')) # Cortos
        m = int(config.get('Seleccion', 'm')) # Duración intermedia
        l = int(config.get('Seleccion', 'l')) # Largos
        # Duración en milisegundos de los límites entre los distintos tipo de ecos:
        # Ecos cortos: < t_l. t_l <= Ecos de duración intermedis < t_h. t_h <= Ecos largos
        t_l = int(config.get('Seleccion', 't_l'))
        t_h = int(config.get('Seleccion', 't_h'))

        # Parámetros para la generación del sonido.
        #Parámetros para el volumen
        Med_1 = float(config.get('Sonidos', 'Med_1'))  # Valor medio
        Amp_1 = float(config.get('Sonidos', 'Amp_1'))  # Amplitud
        #Parámetros para el tono
        Med_2 = float(config.get('Sonidos', 'Med_2'))  # Valor medio
        Amp_2 = float(config.get('Sonidos', 'Amp_2'))  # Amplitud
        # Parámetros básicos del sonido
        basic_freq = [261.6, 523.2, 1046.4] 
        basic_mul = [.3, .3*.9, .3*.5] 

        # Ruta del archivo
        path_file = destination_path + str(datetime.now().strftime("%d_%m_%Y_%H_%M")) + ".wav"
        # Conversión a Dataframes
        tuples = list(zip(t, s_n))

        data = pd.DataFrame(tuples,columns=['time', 'data'])

        var_1 = []  # La variable correspondiente a la curva de luz
        var_2 = []  # La variable correspondiente al efecto Doppler en el espectrograma

        # Extracción, a partir del fichero del espectrograma los valores de la
        # curva de luz y el espectrograma relevantes.
        for i in data['time'].unique():
            var_1.append(data[data['time']==i]['data'].max())


        for i in data['time'].unique():
            l = len(data[data['time']==i]['data'])
            var_2.append(data[data['time']==i]['data'].iloc[int(l*3/8):int(l*5/8)].sum()/len(data[data['time']==i]['data'].iloc[int(l*3/8):int(l*5/8)]))
        # Normalización de los cambios de frecuencia y volumen según los 
        # parámetros establecidos al principio del programa.
        M_1 = max(var_1)
        m_1 = min(var_1)
        if M_1 != m_1:
            a_1 = 2 / (M_1 - m_1)
            b_1 = 1 - 2 * M_1 / (M_1 - m_1)
        else:
            a_1 = 0
            b_1 = 0

        M_2 = max(var_2)
        m_2 = min(var_2)
        if M_2 != m_2:
            a_2 = 2 / (M_2 - m_2)
            b_2 = 1 - 2 * M_2 / (M_2 - m_2)
        else:
            a_2 = 0
            b_2 = 0


        # Creación de las variables normalizadas que se utilizarán para generar
        # los sonidos. La variable 1 se corresponde con la amplitud y la
        # variable 2 con la frecuencia.
        var_1_normalized = [Med_1 + Amp_1 * (a_1 * i + b_1) for i in var_1]
        var_2_normalized = [Med_2 + Amp_2 * (a_2 * i + b_2) for i in var_2]


        inst_freq = [f * var_2_normalized[0] for f in basic_freq]
        inst_mul = [m * var_1_normalized[0] for m in basic_mul]
        # Inicio del servidor para generar el sonido.
        s = Server(duplex=0)
        s.boot()
        s.start()
        # Inicio del proceso de generación del sonido.
        sin = Sine(freq=inst_freq, mul=inst_mul)
        h1 = Harmonizer(sin).out()
        brec = Record(h1, filename=path_file, chnls=2, fileformat=0, sampletype=0)
        clean = Clean_objects(0, brec)

        time.sleep(0.1)

        for i in range(0,len(var_1_normalized)):
            inst_freq = [f * var_2_normalized[i] for f in basic_freq]
            inst_mul = [m * var_1_normalized[i] for m in basic_mul]
            sin.set(attr="freq", value=inst_freq, port=0.05)
            sin.set(attr="mul", value=inst_mul, port=0.05)    
            time.sleep(0.1)

        clean.start()
        s.stop()

        return path_file