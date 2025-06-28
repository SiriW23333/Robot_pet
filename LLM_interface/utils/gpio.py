'''This utill is to check whether the gpio works well.'''
import time
import Hobot.GPIO as GPIO


# GPIO setup
GPIO.setmode(GPIO.BOARD)  # Use BOARD pin numbering
GPIO.setup(11, GPIO.IN)  # Set pin 11 as input
        

if __name__ == "__main__":

    try:
        while True:
            if GPIO.input(11) == 1 :
                print (1)
            elif GPIO.input(11) == 0 :
                print(0)
            time.sleep(1)  # Polling interval
    except KeyboardInterrupt:
        print("Program terminated.")

