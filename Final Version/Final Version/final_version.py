# coding=utf-8
#!/usr/bin/env python3
"""
Scripts to drive a donkey 2 car

Usage:
    manage.py (drive) [--model=<model>] [--js] [--type=(linear|categorical|rnn|imu|behavior|3d|localizer|latent)] [--camera=(single|stereo)] [--meta=<key:value> ...]
    manage.py (train) [--tub=<tub1,tub2,..tubn>] [--file=<file> ...] (--model=<model>) [--transfer=<model>] [--type=(linear|categorical|rnn|imu|behavior|3d|localizer)] [--continuous] [--aug]


Options:
    -h --help          Show this screen.
    --js               Use physical joystick.
    -f --file=<file>   A text file containing paths to tub files, one per line. Option may be used more than once.
    --meta=<key:value> Key/Value strings describing describing a piece of meta data about this drive. Option may be used more than once.
"""
import os
import time
import random

from docopt import docopt
import numpy as np

import donkeycar as dk

### import for new functions
import RPi.GPIO as GPIO
import signal
import atexit
import eventlet


import requests, datetime, time
import threading
import RPi.GPIO as GPIO
from mfrc522 import SimpleMFRC522


from PIL import Image
#import cv2
import glob
import tensorflow as tf
import numpy as np
import predictor

import shutil
path = './data/tub_287_19-09-05'
new_path = './test'
###

#import parts
from donkeycar.parts.transform import Lambda, TriggeredCallback, DelayedTrigger
from donkeycar.parts.datastore import TubHandler
from donkeycar.parts.controller import LocalWebController, JoystickController
from donkeycar.parts.throttle_filter import ThrottleFilter
from donkeycar.parts.behavior import BehaviorPart
from donkeycar.parts.file_watcher import FileWatcher
from donkeycar.parts.launch import AiLaunch


def SG90():
    atexit.register(GPIO.cleanup)
    servopin = 21
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(servopin, GPIO.OUT, initial=False)
    p = GPIO.PWM(servopin, 50)  # 50HZ
    p.start(0)
    time.sleep(2)
    i=0
    while (True):
        # for i in range(0, 181, 10):
        if(i<35):
            p.ChangeDutyCycle(12.0)  # 设置转动角度
            time.sleep(0.02)  # 等该20ms周期结束
            p.ChangeDutyCycle(0)  # 归零信号
            time.sleep(0.1)
            i+=1
        elif(i<76):
        # for i in range(181, 0, -10):
            p.ChangeDutyCycle(2.0)
            time.sleep(0.02)
            p.ChangeDutyCycle(0)
            time.sleep(0.1)
            i+=1
        else:
            break
########################################################################
class MyThread(threading.Thread):
    def __init__(self, target, args=()):
      
        super(MyThread, self).__init__()
        self.func = target
        self.args = args

    def run(self):
        # 接受返回值
        self.result = self.func(*self.args)

    def get_result(self):
        # 线程不结束,返回值为None
        try:
            return self.result
        except Exception:
            return None


# 为了限制真实请求时间或函数执行时间的装饰器
def limit_decor(limit_time):


    def functions(func):
        # 执行操作
        def run(*params):
            thre_func = MyThread(target=func, args=params)
            # 主线程结束(超出时长),则线程方法结束
            thre_func.setDaemon(True)
            thre_func.start()
            # 计算分段沉睡次数
            sleep_num = int(limit_time // 1)
            sleep_nums = round(limit_time % 1, 1)
            # 多次短暂沉睡并尝试获取返回值
            for i in range(sleep_num):
                time.sleep(1)
                infor = thre_func.get_result()
                if infor:
                    return infor
            time.sleep(sleep_nums)
            # 最终返回值(不论线程是否已结束)
            if thre_func.get_result():
                return thre_func.get_result()
            else:
                return ('q','0')  #超时返回  可以自定义

        return run

    return functions

#接口函数
def Read_rfid():
    print("start ")

    #这里把逻辑封装成一个函数,使用线程调用
    a_theadiing = MyThread(target=Read)
    a_theadiing.start()
    a_theadiing.join()

    #返回结果
    id,text = a_theadiing.get_result()

    print("finish")
    return id,text

#################################################################################
@limit_decor(1)   #超时设置为3s   2s逻辑未执行完毕返回接口超时
def Read():
    GPIO.cleanup()
    reader = SimpleMFRC522()
    try:
        id, text = reader.read()
        if(not isinstance(id,str)):
            id='q'
        if(not isinstance(text,str)):
            text='0'
        return (id,text)
    finally:
        GPIO.cleanup()


def predict(new_path):
    frozen_inference_graph_path =  './frozen_inference_graph_pb/'+'frozen_inference_graph.pb'
    images_dir = new_path
    count=0
    model = predictor.Predictor(frozen_inference_graph_path)
    
    image_files = glob.glob(os.path.join(images_dir, '*.*'))

    val_results = []
    predicted_count = 0
    num_samples = len(image_files)
    for image_path in image_files:
        predicted_count += 1
        if predicted_count % 100 == 0:
            print('Predict {}/{}.'.format(predicted_count, num_samples))

        image_name = image_path.split('/')[-1]
        image = Image.open(image_path)
        image = np.asarray(image)
        if image is None:
            print('image %s does not exist.' % image_name)
            continue
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pred_label = int(model.predict([image])[0])
        count=pred_label+count
        print('Image Name: %s' % image_name)
        print('Pred Label: %d' %pred_label)
    return count

def get_images(path,new_path,num):
  delDir = new_path
  delList = os.listdir(delDir)
  for f in delList:
    filePath = os.path.join( delDir, f )
    if os.path.isfile(filePath):
      os.remove(filePath)
 
  MAX=0
  for root, dirs, files in os.walk(path):#返回图片最大值
      for i in range(len(files)):
       if (files[i][-3:] == 'jpg'):
           a = files[i].split('_', 1)[0]
           a = int(a)
           if(MAX<a):
               MAX=a

  MIN=MAX-num+1

  for root, dirs, files in os.walk(path):#保存图片
    for i in range(len(files)):
       if (files[i][-3:] == 'jpg'):
              num=files[i].split('_', 1)[0]
              num=int(num)
              if (num<=MAX and MIN<=num):
                file_path = root + '/' + files[i]
                new_file_path = new_path + '/' + files[i]
                shutil.copy(file_path, new_file_path)










def drive(cfg, model_path=None, use_joystick=False, model_type=None, camera_type='single', meta=[] ):
    '''
    Construct a working robotic vehicle from many parts.
    Each part runs as a job in the Vehicle loop, calling either
    it's run or run_threaded method depending on the constructor flag `threaded`.
    All parts are updated one after another at the framerate given in
    cfg.DRIVE_LOOP_HZ assuming each part finishes processing in a timely manner.
    Parts may have named outputs and inputs. The framework handles passing named outputs
    to parts requesting the same named input.
    '''

    if cfg.DONKEY_GYM:
        #the simulator will use cuda and then we usually run out of resources
        #if we also try to use cuda. so disable for donkey_gym.
        os.environ["CUDA_VISIBLE_DEVICES"]="-1"

    if model_type is None:
        if cfg.TRAIN_LOCALIZER:
            model_type = "localizer"
        elif cfg.TRAIN_BEHAVIORS:
            model_type = "behavior"
        else:
            model_type = cfg.DEFAULT_MODEL_TYPE

    #Initialize car
    V = dk.vehicle.Vehicle()

    if camera_type == "stereo":

        if cfg.CAMERA_TYPE == "WEBCAM":
            from donkeycar.parts.camera import Webcam

            camA = Webcam(image_w=cfg.IMAGE_W, image_h=cfg.IMAGE_H, image_d=cfg.IMAGE_DEPTH, iCam = 0)
            camB = Webcam(image_w=cfg.IMAGE_W, image_h=cfg.IMAGE_H, image_d=cfg.IMAGE_DEPTH, iCam = 1)

        elif cfg.CAMERA_TYPE == "CVCAM":
            from donkeycar.parts.cv import CvCam

            camA = CvCam(image_w=cfg.IMAGE_W, image_h=cfg.IMAGE_H, image_d=cfg.IMAGE_DEPTH, iCam = 0)
            camB = CvCam(image_w=cfg.IMAGE_W, image_h=cfg.IMAGE_H, image_d=cfg.IMAGE_DEPTH, iCam = 1)
        else:
            raise(Exception("Unsupported camera type: %s" % cfg.CAMERA_TYPE))

        V.add(camA, outputs=['cam/image_array_a'], threaded=True)
        V.add(camB, outputs=['cam/image_array_b'], threaded=True)

        from donkeycar.parts.image import StereoPair

        V.add(StereoPair(), inputs=['cam/image_array_a', 'cam/image_array_b'],
            outputs=['cam/image_array'])

    else:

        inputs = []
        threaded = True
        print("cfg.CAMERA_TYPE", cfg.CAMERA_TYPE)
        if cfg.DONKEY_GYM:
            from donkeycar.parts.dgym import DonkeyGymEnv
            cam = DonkeyGymEnv(cfg.DONKEY_SIM_PATH, env_name=cfg.DONKEY_GYM_ENV_NAME)
            threaded = True
            inputs = ['angle', 'throttle']
        elif cfg.CAMERA_TYPE == "PICAM":
            from donkeycar.parts.camera import PiCamera
            cam = PiCamera(image_w=cfg.IMAGE_W, image_h=cfg.IMAGE_H, image_d=cfg.IMAGE_DEPTH)
        elif cfg.CAMERA_TYPE == "WEBCAM":
            from donkeycar.parts.camera import Webcam
            cam = Webcam(image_w=cfg.IMAGE_W, image_h=cfg.IMAGE_H, image_d=cfg.IMAGE_DEPTH)
        elif cfg.CAMERA_TYPE == "CVCAM":
            from donkeycar.parts.cv import CvCam
            cam = CvCam(image_w=cfg.IMAGE_W, image_h=cfg.IMAGE_H, image_d=cfg.IMAGE_DEPTH)
        elif cfg.CAMERA_TYPE == "CSIC":
            from donkeycar.parts.camera import CSICamera
            cam = CSICamera(image_w=cfg.IMAGE_W, image_h=cfg.IMAGE_H, image_d=cfg.IMAGE_DEPTH, framerate=cfg.CAMERA_FRAMERATE)
        elif cfg.CAMERA_TYPE == "V4L":
            from donkeycar.parts.camera import V4LCamera
            cam = V4LCamera(image_w=cfg.IMAGE_W, image_h=cfg.IMAGE_H, image_d=cfg.IMAGE_DEPTH, framerate=cfg.CAMERA_FRAMERATE)
        elif cfg.CAMERA_TYPE == "MOCK":
            from donkeycar.parts.camera import MockCamera
            cam = MockCamera(image_w=cfg.IMAGE_W, image_h=cfg.IMAGE_H, image_d=cfg.IMAGE_DEPTH)
        else:
            raise(Exception("Unkown camera type: %s" % cfg.CAMERA_TYPE))

        V.add(cam, inputs=inputs, outputs=['cam/image_array'], threaded=threaded)

    if use_joystick or cfg.USE_JOYSTICK_AS_DEFAULT:
        #modify max_throttle closer to 1.0 to have more power
        #modify steering_scale lower than 1.0 to have less responsive steering
        from donkeycar.parts.controller import get_js_controller

        ctr = get_js_controller(cfg)

        if cfg.USE_NETWORKED_JS:
            from donkeycar.parts.controller import JoyStickSub
            netwkJs = JoyStickSub(cfg.NETWORK_JS_SERVER_IP)
            V.add(netwkJs, threaded=True)
            ctr.js = netwkJs

    else:
        #This web controller will create a web server that is capable
        #of managing steering, throttle, and modes, and more.
        ctr = LocalWebController()


    V.add(ctr,
          inputs=['cam/image_array'],
          outputs=['user/angle', 'user/throttle', 'user/mode', 'recording'],
          threaded=True)

    #this throttle filter will allow one tap back for esc reverse
    th_filter = ThrottleFilter()
    V.add(th_filter, inputs=['user/throttle'], outputs=['user/throttle'])

    #See if we should even run the pilot module.
    #This is only needed because the part run_condition only accepts boolean
    class PilotCondition:
        def run(self, mode):
            if mode == 'user':
                return False
            else:
                return True

    V.add(PilotCondition(), inputs=['user/mode'], outputs=['run_pilot'])

    class LedConditionLogic:
        def __init__(self, cfg):
            self.cfg = cfg

        def run(self, mode, recording, recording_alert, behavior_state, model_file_changed, track_loc):
            #returns a blink rate. 0 for off. -1 for on. positive for rate.

            if track_loc is not None:
                led.set_rgb(*self.cfg.LOC_COLORS[track_loc])
                return -1

            if model_file_changed:
                led.set_rgb(self.cfg.MODEL_RELOADED_LED_R, self.cfg.MODEL_RELOADED_LED_G, self.cfg.MODEL_RELOADED_LED_B)
                return 0.1
            else:
                led.set_rgb(self.cfg.LED_R, self.cfg.LED_G, self.cfg.LED_B)

            if recording_alert:
                led.set_rgb(*recording_alert)
                return self.cfg.REC_COUNT_ALERT_BLINK_RATE
            else:
                led.set_rgb(self.cfg.LED_R, self.cfg.LED_G, self.cfg.LED_B)

            if behavior_state is not None and model_type == 'behavior':
                r, g, b = self.cfg.BEHAVIOR_LED_COLORS[behavior_state]
                led.set_rgb(r, g, b)
                return -1 #solid on

            if recording:
                return -1 #solid on
            elif mode == 'user':
                return 1
            elif mode == 'local_angle':
                return 0.5
            elif mode == 'local':
                return 0.1
            return 0

    if cfg.HAVE_RGB_LED and not cfg.DONKEY_GYM:
        from donkeycar.parts.led_status import RGB_LED
        led = RGB_LED(cfg.LED_PIN_R, cfg.LED_PIN_G, cfg.LED_PIN_B, cfg.LED_INVERT)
        led.set_rgb(cfg.LED_R, cfg.LED_G, cfg.LED_B)

        V.add(LedConditionLogic(cfg), inputs=['user/mode', 'recording', "records/alert", 'behavior/state', 'modelfile/modified', "pilot/loc"],
              outputs=['led/blink_rate'])

        V.add(led, inputs=['led/blink_rate'])


    def get_record_alert_color(num_records):
        col = (0, 0, 0)
        for count, color in cfg.RECORD_ALERT_COLOR_ARR:
            if num_records >= count:
                col = color
        return col

    class RecordTracker:
        def __init__(self):
            self.last_num_rec_print = 0
            self.dur_alert = 0
            self.force_alert = 0

        def run(self, num_records):
            if num_records is None:
                return 0

            if self.last_num_rec_print != num_records or self.force_alert:
                self.last_num_rec_print = num_records

                '''if num_records % 10 == 0:
                    print("recorded", num_records, "records")'''

                if num_records % cfg.REC_COUNT_ALERT == 0 or self.force_alert:
                    self.dur_alert = num_records // cfg.REC_COUNT_ALERT * cfg.REC_COUNT_ALERT_CYC
                    self.force_alert = 0

            if self.dur_alert > 0:
                self.dur_alert -= 1

            if self.dur_alert != 0:
                return get_record_alert_color(num_records)

            return 0

    rec_tracker_part = RecordTracker()
    V.add(rec_tracker_part, inputs=["tub/num_records"], outputs=['records/alert'])

    if cfg.AUTO_RECORD_ON_THROTTLE and isinstance(ctr, JoystickController):
        #then we are not using the circle button. hijack that to force a record count indication
        def show_record_acount_status():
            rec_tracker_part.last_num_rec_print = 0
            rec_tracker_part.force_alert = 1
        ctr.set_button_down_trigger('circle', show_record_acount_status)

    #Sombrero
    if cfg.HAVE_SOMBRERO:
        from donkeycar.parts.sombrero import Sombrero
        s = Sombrero()

    #IMU
    if cfg.HAVE_IMU:
        from donkeycar.parts.imu import Mpu6050
        imu = Mpu6050()
        V.add(imu, outputs=['imu/acl_x', 'imu/acl_y', 'imu/acl_z',
            'imu/gyr_x', 'imu/gyr_y', 'imu/gyr_z'], threaded=True)

    #Behavioral state
    if cfg.TRAIN_BEHAVIORS:
        bh = BehaviorPart(cfg.BEHAVIOR_LIST)
        V.add(bh, outputs=['behavior/state', 'behavior/label', "behavior/one_hot_state_array"])
        try:
            ctr.set_button_down_trigger('L1', bh.increment_state)
        except:
            pass

        inputs = ['cam/image_array', "behavior/one_hot_state_array"]
    #IMU
    elif model_type == "imu":
        assert(cfg.HAVE_IMU)
        #Run the pilot if the mode is not user.
        inputs=['cam/image_array',
            'imu/acl_x', 'imu/acl_y', 'imu/acl_z',
            'imu/gyr_x', 'imu/gyr_y', 'imu/gyr_z']
    else:
        inputs=['cam/image_array']

    def load_model(kl, model_path):
        start = time.time()
        try:
            print('loading model', model_path)
            kl.load(model_path)
            print('finished loading in %s sec.' % (str(time.time() - start)) )
        except Exception as e:
            print(e)
            print('ERR>> problems loading model', model_path)

    def load_weights(kl, weights_path):
        start = time.time()
        try:
            print('loading model weights', weights_path)
            kl.model.load_weights(weights_path)
            print('finished loading in %s sec.' % (str(time.time() - start)) )
        except Exception as e:
            print(e)
            print('ERR>> problems loading weights', weights_path)

    def load_model_json(kl, json_fnm):
        start = time.time()
        print('loading model json', json_fnm)
        from tensorflow.python import keras
        try:
            with open(json_fnm, 'r') as handle:
                contents = handle.read()
                kl.model = keras.models.model_from_json(contents)
            print('finished loading json in %s sec.' % (str(time.time() - start)) )
        except Exception as e:
            print(e)
            print("ERR>> problems loading model json", json_fnm)

    if model_path:
        #When we have a model, first create an appropriate Keras part
        kl = dk.utils.get_model_by_type(model_type, cfg)

        model_reload_cb = None

        if '.h5' in model_path:
            #when we have a .h5 extension
            #load everything from the model file
            load_model(kl, model_path)

            def reload_model(filename):
                load_model(kl, filename)

            model_reload_cb = reload_model

        elif '.json' in model_path:
            #when we have a .json extension
            #load the model from there and look for a matching
            #.wts file with just weights
            load_model_json(kl, model_path)
            weights_path = model_path.replace('.json', '.weights')
            load_weights(kl, weights_path)

            def reload_weights(filename):
                weights_path = filename.replace('.json', '.weights')
                load_weights(kl, weights_path)

            model_reload_cb = reload_weights

        else:
            print("ERR>> Unknown extension type on model file!!")
            return

        #this part will signal visual LED, if connected
        V.add(FileWatcher(model_path, verbose=True), outputs=['modelfile/modified'])

        #these parts will reload the model file, but only when ai is running so we don't interrupt user driving
        V.add(FileWatcher(model_path), outputs=['modelfile/dirty'], run_condition="ai_running")
        V.add(DelayedTrigger(100), inputs=['modelfile/dirty'], outputs=['modelfile/reload'], run_condition="ai_running")
        V.add(TriggeredCallback(model_path, model_reload_cb), inputs=["modelfile/reload"], run_condition="ai_running")

        outputs=['pilot/angle', 'pilot/throttle']

        if cfg.TRAIN_LOCALIZER:
            outputs.append("pilot/loc")

        V.add(kl, inputs=inputs,
            outputs=outputs,
            run_condition='run_pilot')
#################################################################################################################3

    class Rfid:

        def run(self,rfidtext):
            if(not isinstance(rfidtext,int)):
                rfidtext=0
            if(rfidtext==1 or rfidtext==2 or rfidtext==3):
                return rfidtext

            id,text=Read_rfid()
            if (int(text)==1 or int(text)==2):
                print('text:')
                print(int(text))
                return int(text)
            else:
                print('text:')
                print(0)
                return 0

    V.add(Rfid(), inputs=['rfidtext'], outputs=['rfidtext'])

    class Workcycle:

        def run(self,rfidtext):
            if(rfidtext==0 or rfidtext==3):
                #print('Workcycle:off')
                return 0
            else:
                #print('Workcycle:on')
                return 1
    V.add(Workcycle(), inputs=['rfidtext'], outputs=['workm'])

    class Switch_stop:

        def run(self,throttle,duration,workm):
            judget=0
            if(not isinstance(throttle,float)):
                throttle=0
            if(not isinstance(duration,int)):
                duration=0
            if(workm==0):
                duration=0
                return judget,duration
            if(duration==250 and workm==1):
                judget=1
                duration=0
                print("The car stopped")
                return judget,duration
            if(throttle<0.1 and workm==1):
                duration=duration+1
                #print(duration)

                return judget,duration
            if(throttle>=0.1 and workm==1):
                duration=0
                return judget,duration
    V.add(Switch_stop(), inputs=['throttle','duration','workm'], outputs=['judget','duration'])

    class Work:

        def run(self,judget,workm,rfidtext):
            if (judget==0 and workm==1):
                return judget,rfidtext,1
            if (judget==0 and workm==0):
                return judget,rfidtext,0
            if(workm==1 and judget==1):    
                ###recognition
                ans=1
                time.sleep(1)
                print('Start recognition...')
                get_images(path,new_path,10)
                ans=predict(new_path)
                if(ans>5):
                    ans=2
                else:
                    ans=1
                if (rfidtext==ans):
                    print('Correct box!')
                    time.sleep(1)
                    SG90()
                    time.sleep(2)
                    judget=0
                    rfidtext=3
                    return judget,rfidtext,0
                else:
                    print('Wrong box!')
                    judget=0
                    return judget,rfidtext,1
    V.add(Work(), inputs=['judget','workm','rfidtext'], outputs=['judget','rfidtext','workm'])

    class Workm_x:

        def run(self,workm,flag):
            if(not isinstance(flag,int)):
                flag=0
            if (workm==1):
                flag=1
            if(flag):
                if(workm==0):
                    flag=2
            return flag
    V.add(Workm_x(), inputs=['workm','flag'], outputs=['flag'])

    class Mypilot:

        def run(self):
            return 0,0.25
    V.add(Mypilot(),inputs=[],outputs=['mypilot/angle', 'mypilot/throttle'])

    class Timer:

        def run(self,flag,count):
            judges=0
            if(not isinstance(count,int)):
                count=0
            if(flag==0 or flag==1):
                count=0
                return flag,count,judges
            if(count==20 and flag==2):
                count=0
                flag=0
                judges=0
                return flag,count,judges
            if(flag==2):
                judges=1
                count=count+1
                #print(count)
                return flag,count,judges
    V.add(Timer(),inputs=['flag','count'],outputs=['flag','count','judges'])


 


  



    #Choose what inputs should change the car.
    class DriveMode:
       
        def run(self, mode,
                    user_angle, user_throttle,
                    pilot_angle, pilot_throttle,
                    mypilot_angle, mypilot_throttle,judges):
            if mode == 'user':
                if(judges):
                    return mypilot_angle, mypilot_throttle
                else:
                    return user_angle, user_throttle

            elif mode == 'local_angle':
                return pilot_angle, user_throttle

            else:
                return pilot_angle, pilot_throttle * cfg.AI_THROTTLE_MULT

    V.add(DriveMode(),
          inputs=['user/mode', 'user/angle', 'user/throttle', 'pilot/angle', 'pilot/throttle',
                  'mypilot/angle', 'mypilot/throttle','judges'],
          outputs=['angle', 'throttle'])

#####################################################################################################################
    #to give the car a boost when starting ai mode in a race
  
    aiLauncher = AiLaunch(cfg.AI_LAUNCH_DURATION, cfg.AI_LAUNCH_THROTTLE, cfg.AI_LAUNCH_KEEP_ENABLED)
    V.add(aiLauncher,
        inputs=['user/mode', 'throttle'],
        outputs=['throttle'])

    if isinstance(ctr, JoystickController):
        ctr.set_button_down_trigger(cfg.AI_LAUNCH_ENABLE_BUTTON, aiLauncher.enable_ai_launch)


    class AiRunCondition:
        '''
        A bool part to let us know when ai is running.
        '''
        def run(self, mode):
            if mode == "user":
                return False
            return True

    V.add(AiRunCondition(), inputs=['user/mode'], outputs=['ai_running'])

    #Ai Recording
    class AiRecordingCondition:
        '''
        return True when ai mode, otherwize respect user mode recording flag
        '''
        def run(self, mode, recording):
            if mode == 'user':
                return recording
            return True

    if cfg.RECORD_DURING_AI:
        V.add(AiRecordingCondition(), inputs=['user/mode', 'recording'], outputs=['recording'])

    #Drive train setup
    if cfg.DONKEY_GYM:
        pass

    elif cfg.DRIVE_TRAIN_TYPE == "SERVO_ESC":
        from donkeycar.parts.actuator import PCA9685, PWMSteering, PWMThrottle

        steering_controller = PCA9685(cfg.STEERING_CHANNEL, cfg.PCA9685_I2C_ADDR, busnum=cfg.PCA9685_I2C_BUSNUM)
        steering = PWMSteering(controller=steering_controller,
                                        left_pulse=cfg.STEERING_LEFT_PWM,
                                        right_pulse=cfg.STEERING_RIGHT_PWM)

        throttle_controller = PCA9685(cfg.THROTTLE_CHANNEL, cfg.PCA9685_I2C_ADDR, busnum=cfg.PCA9685_I2C_BUSNUM)
        throttle = PWMThrottle(controller=throttle_controller,
                                        max_pulse=cfg.THROTTLE_FORWARD_PWM,
                                        zero_pulse=cfg.THROTTLE_STOPPED_PWM,
                                        min_pulse=cfg.THROTTLE_REVERSE_PWM)

        V.add(steering, inputs=['angle'])
        V.add(throttle, inputs=['throttle'])


    elif cfg.DRIVE_TRAIN_TYPE == "DC_STEER_THROTTLE":
        from donkeycar.parts.actuator import Mini_HBridge_DC_Motor_PWM

        steering = Mini_HBridge_DC_Motor_PWM(cfg.HBRIDGE_PIN_LEFT, cfg.HBRIDGE_PIN_RIGHT)
        throttle = Mini_HBridge_DC_Motor_PWM(cfg.HBRIDGE_PIN_FWD, cfg.HBRIDGE_PIN_BWD)

        V.add(steering, inputs=['angle'])
        V.add(throttle, inputs=['throttle'])


    elif cfg.DRIVE_TRAIN_TYPE == "DC_TWO_WHEEL":
        from donkeycar.parts.actuator import TwoWheelSteeringThrottle, Mini_HBridge_DC_Motor_PWM

        left_motor = Mini_HBridge_DC_Motor_PWM(cfg.HBRIDGE_PIN_LEFT_FWD, cfg.HBRIDGE_PIN_LEFT_BWD)
        right_motor = Mini_HBridge_DC_Motor_PWM(cfg.HBRIDGE_PIN_RIGHT_FWD, cfg.HBRIDGE_PIN_RIGHT_BWD)
        two_wheel_control = TwoWheelSteeringThrottle()

        V.add(two_wheel_control,
                inputs=['throttle', 'angle'],
                outputs=['left_motor_speed', 'right_motor_speed'])

        V.add(left_motor, inputs=['left_motor_speed'])
        V.add(right_motor, inputs=['right_motor_speed'])

    elif cfg.DRIVE_TRAIN_TYPE == "SERVO_HBRIDGE_PWM":
        from donkeycar.parts.actuator import ServoBlaster, PWMSteering
        steering_controller = ServoBlaster(cfg.STEERING_CHANNEL) #really pin
        #PWM pulse values should be in the range of 100 to 200
        assert(cfg.STEERING_LEFT_PWM <= 200)
        assert(cfg.STEERING_RIGHT_PWM <= 200)
        steering = PWMSteering(controller=steering_controller,
                                        left_pulse=cfg.STEERING_LEFT_PWM,
                                        right_pulse=cfg.STEERING_RIGHT_PWM)


        from donkeycar.parts.actuator import Mini_HBridge_DC_Motor_PWM
        motor = Mini_HBridge_DC_Motor_PWM(cfg.HBRIDGE_PIN_FWD, cfg.HBRIDGE_PIN_BWD)

        V.add(steering, inputs=['angle'])
        V.add(motor, inputs=["throttle"])


    #add tub to save data

    inputs=['cam/image_array',
            'user/angle', 'user/throttle',
            'user/mode']

    types=['image_array',
           'float', 'float',
           'str']

    if cfg.TRAIN_BEHAVIORS:
        inputs += ['behavior/state', 'behavior/label', "behavior/one_hot_state_array"]
        types += ['int', 'str', 'vector']

    if cfg.HAVE_IMU:
        inputs += ['imu/acl_x', 'imu/acl_y', 'imu/acl_z',
            'imu/gyr_x', 'imu/gyr_y', 'imu/gyr_z']

        types +=['float', 'float', 'float',
           'float', 'float', 'float']

    if cfg.RECORD_DURING_AI:
        inputs += ['pilot/angle', 'pilot/throttle']
        types += ['float', 'float']

    th = TubHandler(path=cfg.DATA_PATH)
    tub = th.new_tub_writer(inputs=inputs, types=types, user_meta=meta)
    V.add(tub, inputs=inputs, outputs=["tub/num_records"], run_condition='recording')

    if cfg.PUB_CAMERA_IMAGES:
        from donkeycar.parts.network import TCPServeValue
        from donkeycar.parts.image import ImgArrToJpg
        pub = TCPServeValue("camera")
        V.add(ImgArrToJpg(), inputs=['cam/image_array'], outputs=['jpg/bin'])
        V.add(pub, inputs=['jpg/bin'])

    if type(ctr) is LocalWebController:
        print("You can now go to <your pi ip address>:8887 to drive your car.")
    elif isinstance(ctr, JoystickController):
        print("You can now move your joystick to drive your car.")
        #tell the controller about the tub
        ctr.set_tub(tub)

        if cfg.BUTTON_PRESS_NEW_TUB:

            def new_tub_dir():
                V.parts.pop()
                tub = th.new_tub_writer(inputs=inputs, types=types, user_meta=meta)
                V.add(tub, inputs=inputs, outputs=["tub/num_records"], run_condition='recording')
                ctr.set_tub(tub)

            ctr.set_button_down_trigger('cross', new_tub_dir)
        ctr.print_controls()

    #run the vehicle for 20 seconds
    V.start(rate_hz=cfg.DRIVE_LOOP_HZ,
            max_loop_count=cfg.MAX_LOOPS)


if __name__ == '__main__':
    args = docopt(__doc__)
    cfg = dk.load_config()

    if args['drive']:
        model_type = args['--type']
        camera_type = args['--camera']
        drive(cfg, model_path=args['--model'], use_joystick=args['--js'], model_type=model_type, camera_type=camera_type,
            meta=args['--meta'])

    if args['train']:
        from train import multi_train, preprocessFileList

        tub = args['--tub']
        model = args['--model']
        transfer = args['--transfer']
        model_type = args['--type']
        continuous = args['--continuous']
        aug = args['--aug']

        dirs = preprocessFileList( args['--file'] )
        if tub is not None:
            tub_paths = [os.path.expanduser(n) for n in tub.split(',')]
            dirs.extend( tub_paths )

        multi_train(cfg, dirs, model, transfer, model_type, continuous, aug)

