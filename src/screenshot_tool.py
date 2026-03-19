import cv2
import numpy as np
import pyautogui
import keyboard  # muss per pip install keyboard installiert werden
import os
import time

# Ensure utf-8 output to avoid charmap encode errors on Windows
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')


SAVE_PATH = "data/raw/"
os.makedirs(SAVE_PATH, exist_ok=True)

print("Tool gestartet. Drücke 'F10' für einen Screenshot, 'ESC' zum Beenden.")

count = 0
while True:
    if keyboard.is_pressed('f10'):
        # Screenshot machen
        img = pyautogui.screenshot()
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        
        # Speichern
        timestamp = int(time.time())
        filename = f"{SAVE_PATH}shot_{timestamp}_{count}.png"
        cv2.imwrite(filename, img)
        
        print(f"Gespeichert: {filename}")
        count += 1
        time.sleep(0.5) # Verhindert Mehrfach-Auslösung
        
    if keyboard.is_pressed('esc'):
        break