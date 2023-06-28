from datetime import datetime
import time
import statistics as stats
import csv


class PerfLog:    
    logger = {}

    def __init__(self, filename, runname):
        self.filename = filename
        self.runname = runname

    def log(self, msg: str, elapsedMs ):
        print(msg, elapsedMs, "ms") 
        with open(self.filename, mode='a') as file:
            writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL )
            # format:         
            writer.writerow([ datetime.now(), self.runname, msg, elapsedMs ])
    
    @staticmethod
    def getLogger(filename: str, name: str):
        if name in PerfLog.logger:
            return PerfLog.logger[name]
        p = PerfLog(filename, name)
        PerfLog.logger[name] = p    
        return p


class PerfTimer:
    def __init__(self, l: PerfLog) -> None:
        self.plog = l
        
    def start(self, section: str=""):
        self.start = time.time_ns()
        self.section = section
        return self.start

    def log(self, msg: str=""):
        elapsedMs = (time.time_ns() - self.start) / 1000000        
        self.plog.log(self.section + ': ' + msg, elapsedMs)           
        return elapsedMs