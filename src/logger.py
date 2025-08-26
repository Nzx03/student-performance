import logging #any excetution happening is get stored in log file
import os
from datetime import datetime

LOG_FILE=f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"  #this willbe txt file

logs_path=os.path.join(os.getcwd(),"logs",LOG_FILE)   #for getting working directory
os.makedirs(logs_path,exist_ok=True)


LOG_FILE_PATH=os.path.join(logs_path,LOG_FILE)

logging.basicConfig(filename=LOG_FILE_PATH,
                    format="[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s",
                    level=logging.INFO,
                    
                    
                    
                    
                    )


