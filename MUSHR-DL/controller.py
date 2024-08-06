import time
import traceback
import numpy as np
from key_check import *


def check_range(x,lim):
    if(x>lim):
        return lim
    elif(x<-lim):
        return -lim
    return x

class keyboard_control():
    def __init__(self):
        self.st = 0.0
        self.th = 0.0
        self.autonomous = True
        # reset_mouse_pos()

    def read_loop(self):
        print("Set default as manual mode")
        while 1:
            try:
                time.sleep(0.01)
                # self.st,self.th = get_mouse_pos()
                if(key_press() == ['A']):
                    print("Change to auto mode")
                    self.st = 0.0
                    self.th = 0.0
                    self.autonomous = True
                if(key_press() == ['M']):
                    print("Change to manual mode")
                    self.st = 0.0
                    self.th = 0.0
                    self.autonomous = False
                if(key_press() == ['E']):
                    self.th += 0.01
                elif(key_press() == ['D']):
                    self.th -= 0.01
                elif(key_press() == ['S']):
                    self.st -= 0.01
                elif(key_press() == ['F']):
                    self.st += 0.01
                else:
                    if self.st > 0.0:
                        self.st -= 0.01
                    elif self.st < 0.0:
                        self.st += 0.01

                self.st = check_range(self.st,1.0)
                self.th = check_range(self.th,1.0)
                # print(self.st, self.th)
            except KeyboardInterrupt:
                exit()
            except Exception as e:
                # pass
                print(traceback.format_exc())
            except:
                pass

if __name__ == "__main__": 
    obj = keyboard_control()
    obj.read_loop()
